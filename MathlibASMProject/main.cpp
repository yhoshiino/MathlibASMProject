// Benchmark nanobench : traitements par lots (dot / normalize / transform) en plusieurs versions
// + fonctions isolees (Vector3, Mat4x4, scalaire vs simd, complement hors cahier).
//
// Utilisation :  Benchmark.exe [traitement] [version] [tailles...]
//   traitement : all | dot | normalize | transform | vector3 | matrix | check
//   version    : all | novec | sse | soa | asm    (la reference "ref" est TOUJOURS mesuree : elle sert de base a "relative")
//                "asm" n'existe que pour dot ; "soa" mesure conversions, calcul seul et total.
//   tailles    : n1 n2 ...   (defaut : 16 4096 1048576)
// Exemples :  Benchmark.exe                         -> tout, 3 tailles
//             Benchmark.exe dot                     -> dot, toutes versions
//             Benchmark.exe normalize soa 4096      -> normalize : ref + SoA (conversions comprises), n = 4096
//             Benchmark.exe check                   -> verifications seules
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <functional>
#include <random>
#include <utility>
#include <string>
#include <vector>

#include "../MathLibrary/Nanobench/nanobench.h"
#include "../MathLibrary/BatchOps.h"        // inclut Vector3SIMD.h, Matrix4x4SIMD.h
#include "../MathLibrary/Vector3.h"         // version scalaire (math::scalar)
#include "../MathLibrary/Matrice4x4.h"      // version scalaire (math::scalar)

using ankerl::nanobench::Bench;
using ankerl::nanobench::doNotOptimizeAway;
using Measure = ankerl::nanobench::Result::Measure;
using math::simd::Mat4x4f;
using math::simd::Vector3;
using ScalarVec3 = math::scalar::Vector3<float>;
using ScalarMat4 = math::scalar::Mat4x4<float>;
namespace batch = math::batch;

static constexpr unsigned kSeed = 42;
static constexpr size_t   kWarmup = 20;
static constexpr size_t   kEpochs = 21;

static float randomFloat(ankerl::nanobench::Rng& rng) { return float(rng.uniform01()) * 20.0f - 10.0f; }
static Vector3 randomVec(ankerl::nanobench::Rng& rng) {
    float x = randomFloat(rng), y = randomFloat(rng), z = randomFloat(rng);
    return Vector3(x, y, z);
}

// ============================================================================
// Vérifications exécutées AVANT toute mesure (valeurs connues, référence double, bit-exact, tailles limites)
// ============================================================================
using namespace math::batch;

namespace {

    // ---- Bornes d'erreur (modèle standard de l'arithmétique flottante, arrondi au plus proche) ----
    // u = 2^-24 = erreur relative max d'UNE opération float.
    // dot (3 produits, 2 additions)  : |err| <= gamma_3 * Σ|x_i y_i|, gamma_3 ~ 3u  -> borne retenue 4u (marge).
    // normalize : len2 ~3u, sqrt ~1.5u+u, division u -> ~3.5u relatif par composante -> borne 8u.
    // transform (4 termes)           : |err| <= gamma_4 * Σ|termes|, gamma_4 ~ 4u    -> borne retenue 5u.
    constexpr double kU = 5.9604644775390625e-8;
    constexpr double kDotBound = 4.0 * kU;
    constexpr double kNormBound = 8.0 * kU;
    constexpr double kTransBound = 5.0 * kU;
    constexpr std::size_t kGuard = 8;                 // éléments sentinelles de chaque côté

    int g_fail = 0;
    void expect(bool ok, const char* what, const char* impl) {
        if (!ok) { ++g_fail; std::printf("  ECHEC : %s [%s]\n", what, impl); }
    }
    bool approx(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

    // ---- Listes des versions (la première est toujours la référence) ----
    using DotFn = std::function<void(const Vector3*, const Vector3*, float*, std::size_t)>;
    using NormFn = std::function<void(const Vector3*, Vector3*, std::size_t)>;
    using TrFn = std::function<void(const Mat4x4f&, const Vector3*, Vector3*, std::size_t)>;

    std::vector<std::pair<const char*, DotFn>> dotImpls() {
        return {
            {"ref", dot_ref}, {"novec", dot_ref_novec}, {"sse", dot_sse},
            {"soa", [](const Vector3* a, const Vector3* b, float* o, std::size_t n) {
                Vec3SoA sa(n), sb(n); aos_to_soa(a, sa, n); aos_to_soa(b, sb, n); dot_soa_sse(sa, sb, o, n); }},
            {"asm", [](const Vector3* a, const Vector3* b, float* o, std::size_t n) { dot_batch_asm(a, b, o, n); }},
            {"asm1", [](const Vector3* a, const Vector3* b, float* o, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i) o[i] = dot_vec3_asm(&a[i].x, &b[i].x); }},
        };
    }
    std::vector<std::pair<const char*, NormFn>> normImpls() {
        return {
            {"ref", normalize_ref}, {"novec", normalize_ref_novec}, {"sse", normalize_sse},
            {"soa", [](const Vector3* in, Vector3* o, std::size_t n) {
                Vec3SoA si(n), so(n); aos_to_soa(in, si, n); normalize_soa_sse(si, so, n); soa_to_aos(so, o, n); }},
        };
    }
    std::vector<std::pair<const char*, TrFn>> trImpls() {
        return {
            {"ref", transform_ref}, {"novec", transform_ref_novec}, {"sse", transform_sse},
            {"soa", [](const Mat4x4f& m, const Vector3* in, Vector3* o, std::size_t n) {
                Vec3SoA si(n), so(n); aos_to_soa(in, si, n); transform_soa_sse(m, si, so, n); soa_to_aos(so, o, n); }},
        };
    }

    // ------------------------------------------------------------------------
    // 1. Valeurs connues
    // ------------------------------------------------------------------------
    void checkKnown() {
        // dot : (1,2,3).(4,5,6) = 32 ; orthogonaux = 0 ; avec vecteur nul = 0 ; signe négatif
        {
            const Vector3 a[4] = { Vector3(1.f,2.f,3.f), Vector3(1.f,0.f,0.f), Vector3(0.f,0.f,0.f), Vector3(-1.f,-2.f,-3.f) };
            const Vector3 b[4] = { Vector3(4.f,5.f,6.f), Vector3(0.f,1.f,0.f), Vector3(7.f,8.f,9.f), Vector3(4.f,5.f,6.f) };
            const float exp[4] = { 32.f, 0.f, 0.f, -32.f };
            for (auto& [name, f] : dotImpls()) {
                float out[4] = {};
                f(a, b, out, 4);
                for (int i = 0; i < 4; ++i) expect(out[i] == exp[i], "dot valeurs connues", name);
                f(nullptr, nullptr, nullptr, 0);                       // lot vide : rien n'est lu ni écrit
            }
        }
        // normalize : (3,4,0)->(0.6,0.8,0) ; nul->0 ; (0,0,5)->(0,0,1) ; |v|^2 <= 1e-12 -> 0 ; NaN -> 0
        {
            const Vector3 in[5] = { Vector3(3.f,4.f,0.f), Vector3(0.f,0.f,0.f), Vector3(0.f,0.f,5.f),
                                    Vector3(1e-7f,0.f,0.f), Vector3(std::nanf(""),0.f,0.f) };
            const float exp[5][3] = { {0.6f,0.8f,0.f}, {0,0,0}, {0,0,1.f}, {0,0,0}, {0,0,0} };
            for (auto& [name, f] : normImpls()) {
                Vector3 out[5];
                f(in, out, 5);
                for (int i = 0; i < 5; ++i)
                    expect(out[i].x == exp[i][0] && out[i].y == exp[i][1] && out[i].z == exp[i][2] && out[i].w == 0.f,
                        "normalize valeurs connues (nul, seuil, NaN, w=0)", name);
                f(nullptr, nullptr, 0);
            }
        }
        // transform : identité, translation, rotation 90°, composition T*R et R*T (ordre d'application)
        {
            const float halfPi = 1.57079632679f;
            const Vector3 p[3] = { Vector3(1.f,2.f,3.f), Vector3(0.f,0.f,0.f), Vector3(-5.f,6.f,7.f) };
            const Mat4x4f I;
            const Mat4x4f T = Mat4x4f::translate(1.f, 2.f, 3.f);
            const Mat4x4f R = Mat4x4f::rotationZ(halfPi);
            const Vector3 ex(1.f, 0.f, 0.f);
            for (auto& [name, f] : trImpls()) {
                Vector3 out[3];
                f(I, p, out, 3);                                        // identité : exact
                for (int i = 0; i < 3; ++i) expect(out[i].x == p[i].x && out[i].y == p[i].y && out[i].z == p[i].z, "transform identite", name);
                f(T, p, out, 3);                                        // translation : exact
                for (int i = 0; i < 3; ++i) expect(out[i].x == p[i].x + 1.f && out[i].y == p[i].y + 2.f && out[i].z == p[i].z + 3.f, "transform translation", name);
                Vector3 o1;
                f(R, &ex, &o1, 1);                                      // (1,0,0) -> (0,1,0)
                expect(approx(o1.x, 0.f, 1e-6f) && approx(o1.y, 1.f, 1e-6f) && approx(o1.z, 0.f, 1e-6f), "transform rotationZ 90deg", name);
                f(T * R, &ex, &o1, 1);                                  // R d'abord puis T : (0,1,0)+(1,2,3) = (1,3,3)
                expect(approx(o1.x, 1.f, 1e-6f) && approx(o1.y, 3.f, 1e-6f) && approx(o1.z, 3.f, 1e-6f), "composition T*R", name);
                f(R * T, &ex, &o1, 1);                                  // T d'abord : (2,2,3) puis R : (-2,2,3)
                expect(approx(o1.x, -2.f, 1e-6f) && approx(o1.y, 2.f, 1e-6f) && approx(o1.z, 3.f, 1e-6f), "composition R*T", name);
                f(I, nullptr, nullptr, 0);
            }
        }
        std::printf("  valeurs connues        : %s (dot, normalize nul/seuil/NaN, identite, translation, rotation, T*R, R*T, lot vide)\n",
            g_fail == 0 ? "OK" : "ECHEC");
    }

    // ------------------------------------------------------------------------
    // 2 + 3. Référence double + égalité bit à bit
    // ------------------------------------------------------------------------
    void checkRandom(std::size_t n) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> d(-10.f, 10.f);
        std::vector<Vector3> a(n), b(n);
        for (auto& v : a) { float x = d(gen), y = d(gen), z = d(gen); v = Vector3(x, y, z); }
        for (auto& v : b) { float x = d(gen), y = d(gen), z = d(gen); v = Vector3(x, y, z); }
        const Mat4x4f m = Mat4x4f::translate(1.5f, -2.0f, 3.0f) * Mat4x4f::rotationZ(0.7f);
        const double* dummy = nullptr; (void)dummy;

        // --- dot ---
        {
            auto impls = dotImpls();
            std::vector<float> first; bool exact = true; double worst = 0.0;
            for (auto& [name, f] : impls) {
                std::vector<float> o(n); f(a.data(), b.data(), o.data(), n);
                if (first.empty()) first = o; else exact &= (std::memcmp(o.data(), first.data(), n * sizeof(float)) == 0);
                for (std::size_t i = 0; i < n; ++i) {
                    const double t0 = (double)a[i].x * b[i].x, t1 = (double)a[i].y * b[i].y, t2 = (double)a[i].z * b[i].z;
                    const double bound = kDotBound * (std::fabs(t0) + std::fabs(t1) + std::fabs(t2));
                    worst = std::max(worst, std::fabs((double)o[i] - (t0 + t1 + t2)) / (bound + 1e-300));
                }
            }
            expect(exact, "dot : versions identiques bit a bit", "toutes");
            expect(worst <= 1.0, "dot : erreur <= borne vs double", "toutes");
            std::printf("  dot       n=%-6zu : bit-exact entre %zu versions : %s | erreur max vs double = %.2f x borne (4u.sum|x_i y_i|)\n",
                n, impls.size(), exact ? "OUI" : "NON", worst);
        }
        // --- normalize ---
        {
            auto impls = normImpls();
            std::vector<Vector3> first; bool exact = true; double worst = 0.0;
            for (auto& [name, f] : impls) {
                std::vector<Vector3> o(n); f(a.data(), o.data(), n);
                if (first.empty()) first = o; else exact &= (std::memcmp(o.data(), first.data(), n * sizeof(Vector3)) == 0);
                for (std::size_t i = 0; i < n; ++i) {
                    const double x = a[i].x, y = a[i].y, z = a[i].z, len = std::sqrt(x * x + y * y + z * z);
                    const double ex[3] = { x / len, y / len, z / len };
                    const float got[3] = { o[i].x, o[i].y, o[i].z };
                    for (int c = 0; c < 3; ++c)
                        worst = std::max(worst, std::fabs((double)got[c] - ex[c]) / (kNormBound * std::fabs(ex[c]) + 1e-300));
                }
            }
            expect(exact, "normalize : versions identiques bit a bit", "toutes");
            expect(worst <= 1.0, "normalize : erreur <= borne vs double", "toutes");
            std::printf("  normalize n=%-6zu : bit-exact entre %zu versions : %s | erreur max vs double = %.2f x borne (8u relatif)\n",
                n, impls.size(), exact ? "OUI" : "NON", worst);
        }
        // --- transform ---
        {
            auto impls = trImpls();
            const float* md = m.data;
            std::vector<Vector3> first; bool exact = true; double worst = 0.0;
            for (auto& [name, f] : impls) {
                std::vector<Vector3> o(n); f(m, a.data(), o.data(), n);
                if (first.empty()) first = o; else exact &= (std::memcmp(o.data(), first.data(), n * sizeof(Vector3)) == 0);
                for (std::size_t i = 0; i < n; ++i) {
                    const double x = a[i].x, y = a[i].y, z = a[i].z;
                    const float got[3] = { o[i].x, o[i].y, o[i].z };
                    for (int r = 0; r < 3; ++r) {
                        const double t0 = md[r] * x, t1 = md[4 + r] * y, t2 = md[8 + r] * z, t3 = md[12 + r];
                        const double bound = kTransBound * (std::fabs(t0) + std::fabs(t1) + std::fabs(t2) + std::fabs(t3));
                        worst = std::max(worst, std::fabs((double)got[r] - (t0 + t1 + t2 + t3)) / (bound + 1e-300));
                    }
                }
            }
            expect(exact, "transform : versions identiques bit a bit", "toutes");
            expect(worst <= 1.0, "transform : erreur <= borne vs double", "toutes");
            std::printf("  transform n=%-6zu : bit-exact entre %zu versions : %s | erreur max vs double = %.2f x borne (5u.sum|termes|)\n",
                n, impls.size(), exact ? "OUI" : "NON", worst);
        }
    }

    // ------------------------------------------------------------------------
    // 4. Tailles limites : zones sentinelles + résultats = ref
    // ------------------------------------------------------------------------
    template <class T>
    bool guardsIntact(const std::vector<T>& buf, std::size_t n) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(buf.data());
        const std::size_t front = kGuard * sizeof(T), backStart = (kGuard + n) * sizeof(T), total = buf.size() * sizeof(T);
        for (std::size_t i = 0; i < front; ++i) if (p[i] != 0xA5) return false;
        for (std::size_t i = backStart; i < total; ++i) if (p[i] != 0xA5) return false;
        return true;
    }

    void checkBounds() {
        const std::size_t sizes[] = { 0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 1000 };
        std::mt19937 gen(7);
        std::uniform_real_distribution<float> d(-10.f, 10.f);
        const Mat4x4f m = Mat4x4f::translate(1.5f, -2.0f, 3.0f) * Mat4x4f::rotationZ(0.7f);
        const int before = g_fail;

        for (std::size_t n : sizes) {
            std::vector<Vector3> a(n), b(n);
            for (auto& v : a) { float x = d(gen), y = d(gen), z = d(gen); v = Vector3(x, y, z); }
            for (auto& v : b) { float x = d(gen), y = d(gen), z = d(gen); v = Vector3(x, y, z); }

            { // dot
                std::vector<float> refBuf;
                for (auto& [name, f] : dotImpls()) {
                    std::vector<float> buf(n + 2 * kGuard);
                    std::memset(buf.data(), 0xA5, buf.size() * sizeof(float));
                    f(a.data(), b.data(), buf.data() + kGuard, n);
                    expect(guardsIntact(buf, n), "dot : ecriture hors [0,n)", name);
                    if (refBuf.empty()) refBuf = buf;
                    else expect(std::memcmp(buf.data(), refBuf.data(), buf.size() * sizeof(float)) == 0, "dot : resultat != ref", name);
                }
            }
            { // normalize
                std::vector<Vector3> refBuf;
                for (auto& [name, f] : normImpls()) {
                    std::vector<Vector3> buf(n + 2 * kGuard);
                    std::memset((void*)buf.data(), 0xA5, buf.size() * sizeof(Vector3));
                    f(a.data(), buf.data() + kGuard, n);
                    expect(guardsIntact(buf, n), "normalize : ecriture hors [0,n)", name);
                    if (refBuf.empty()) refBuf = buf;
                    else expect(std::memcmp((void*)buf.data(), (void*)refBuf.data(), buf.size() * sizeof(Vector3)) == 0, "normalize : resultat != ref", name);
                }
            }
            { // transform
                std::vector<Vector3> refBuf;
                for (auto& [name, f] : trImpls()) {
                    std::vector<Vector3> buf(n + 2 * kGuard);
                    std::memset((void*)buf.data(), 0xA5, buf.size() * sizeof(Vector3));
                    f(m, a.data(), buf.data() + kGuard, n);
                    expect(guardsIntact(buf, n), "transform : ecriture hors [0,n)", name);
                    if (refBuf.empty()) refBuf = buf;
                    else expect(std::memcmp((void*)buf.data(), (void*)refBuf.data(), buf.size() * sizeof(Vector3)) == 0, "transform : resultat != ref", name);
                }
            }
            { // aller-retour AoS -> SoA -> AoS
                Vec3SoA s(n); std::vector<Vector3> back(n);
                aos_to_soa(a.data(), s, n); soa_to_aos(s, back.data(), n);
                expect(std::memcmp((void*)back.data(), (void*)a.data(), n * sizeof(Vector3)) == 0, "aller-retour AoS->SoA->AoS", "conv");
            }
        }
        std::printf("  tailles limites        : %s (n = 0,1,2,3,4,5,7,8,9,15,16,17,31,1000 ; sentinelles intactes, resultats = ref, aller-retour SoA)\n",
            g_fail == before ? "OK" : "ECHEC");
        std::printf("                           (les lectures hors bornes ne sont pas detectables ici : voir /fsanitize=address)\n");
    }

} // namespace

bool runSelfChecks() {
    g_fail = 0;
    std::printf("\n--- Verifications (avant mesures) ---\n");
    checkKnown();
    checkRandom(4099);                 // 4099 = 4*1024 + 3 : exerce le traitement du reste
    checkBounds();
    std::printf("--- %s ---\n", g_fail == 0 ? "Toutes les verifications passent" : "VERIFICATIONS EN ECHEC");
    return g_fail == 0;
}

// ============================================================================
// Environnement (CPU, caches, compilateur) : exige par le cahier des charges
// ============================================================================
static std::string cpuBrand() {
    int r[4]; char b[49] = {};
    __cpuid(r, 0x80000000);
    if (unsigned(r[0]) < 0x80000004u) return "inconnu";
    for (int i = 0; i < 3; ++i) { __cpuid(r, 0x80000002 + i); std::memcpy(b + 16 * i, r, 16); }
    return b;
}

static void printEnv() {
    std::printf("CPU : %s\n", cpuBrand().c_str());

    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> info(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (len && GetLogicalProcessorInformation(info.data(), &len)) {
        size_t l1 = 0, l2 = 0, l3 = 0;
        for (const auto& i : info) {
            if (i.Relationship != RelationCache) continue;
            const auto& c = i.Cache;
            if (c.Level == 1 && c.Type != CacheInstruction) l1 = (std::max)(l1, size_t(c.Size));
            if (c.Level == 2) l2 = (std::max)(l2, size_t(c.Size));
            if (c.Level == 3) l3 = (std::max)(l3, size_t(c.Size));
        }
        std::printf("Caches (par cache) : L1d %zu Ko | L2 %zu Ko | L3 %.1f Mo\n", l1 / 1024, l2 / 1024, l3 / 1048576.0);
    }
#ifdef _MSC_VER
    std::printf("Compilateur : MSVC %d, C++ (_MSVC_LANG = %ld)\n", _MSC_VER, long(_MSVC_LANG));
#endif
#if defined(_DEBUG)
    std::printf("!!! ATTENTION : build DEBUG, mesures invalides. Passer en Release x64 (Ctrl+F5) !!!\n");
#elif defined(NDEBUG)
    std::printf("Build : Release (NDEBUG)");
#else
    std::printf("Build : NDEBUG non defini");
#endif
#if defined(__AVX2__)
    std::printf(" | jeu d'instructions : /arch:AVX2");
#elif defined(__AVX__)
    std::printf(" | jeu d'instructions : /arch:AVX");
#else
    std::printf(" | jeu d'instructions : SSE2 (defaut x64), intrinsics SSE/SSE2 uniquement");
#endif
#if defined(_M_FP_FAST)
    std::printf(" | flottants : /fp:fast\n");
#elif defined(_M_FP_STRICT)
    std::printf(" | flottants : /fp:strict\n");
#else
    std::printf(" | flottants : /fp:precise\n");
#endif
    std::printf("Options non detectables a l'execution (/O2, /GL, /Oi...) : a copier depuis Proprietes > C/C++ > Ligne de commande.\n");
}

// CPU hybride (P-cores + E-cores) : on epingle le thread sur un P-core pour eviter les migrations
// (cause du bruit sur les grands lots). Sur un CPU non hybride, rien n'est modifie.
static void pinToPerformanceCore() {
    ULONG len = 0;
    GetSystemCpuSetInformation(nullptr, 0, &len, GetCurrentProcess(), 0);
    if (!len) return;
    std::vector<BYTE> buf(len);
    if (!GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data()), len, &len, GetCurrentProcess(), 0)) return;

    BYTE minE = 255, maxE = 0;
    for (ULONG off = 0; off < len;) {
        auto* e = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data() + off);
        if (e->Type == CpuSetInformation) {
            minE = (std::min)(minE, e->CpuSet.EfficiencyClass);
            maxE = (std::max)(maxE, e->CpuSet.EfficiencyClass);
        }
        off += e->Size;
    }
    if (minE == maxE) { std::printf("CPU non hybride : thread non epingle.\n"); return; }

    for (ULONG off = 0; off < len;) {
        auto* e = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data() + off);
        if (e->Type == CpuSetInformation && e->CpuSet.Group == 0 && e->CpuSet.EfficiencyClass == maxE) {
            const unsigned idx = e->CpuSet.LogicalProcessorIndex;
            if (SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << idx))
                std::printf("CPU hybride : thread epingle sur le coeur logique %u (P-core, EfficiencyClass %u)\n", idx, unsigned(maxE));
            return;
        }
        off += e->Size;
    }
}

// ============================================================================
// Affichage : relative | ns/<unite> | err% | total | benchmark
//   relative = temps de la ligne de base / temps de la ligne (100 % = base, 300 % = 3x plus rapide)
//   err%     = erreur absolue mediane en pourcent (dispersion des 21 epoques), total = duree cumulee de la mesure
//   pairs = false : base = 1re ligne du tableau ; pairs = true : base = ligne paire precedente (ref, optimise)
// ============================================================================
static Bench makeBench(size_t batchSize, size_t warmup, const char* unit) {
    Bench b;
    b.output(nullptr)
        .unit(unit)
        .batch(batchSize)
        .warmup(warmup)
        .epochs(kEpochs)
        .minEpochIterations(10)
        .minEpochTime(std::chrono::milliseconds(10))
        .performanceCounters(false);
    return b;
}

static void printTable(const std::string& title, const Bench& bench, const char* unit, bool pairs) {
    const auto& res = bench.results();
    std::printf("\n%s\n", title.c_str());
    std::printf("| %10s | %12s | %7s | %10s | %s\n", "relative", (std::string("ns/") + unit).c_str(), "err%", "total", "benchmark");
    std::printf("|-----------:|-------------:|--------:|-----------:|:-----------------\n");
    for (size_t i = 0; i < res.size(); ++i) {
        const auto& r = res[i];
        const auto& base = pairs ? res[i & ~size_t(1)] : res[0];
        const double ns = r.median(Measure::elapsed) / r.config().mBatch * 1e9;
        const double nsBase = base.median(Measure::elapsed) / base.config().mBatch * 1e9;
        const double err = r.medianAbsolutePercentError(Measure::elapsed) * 100.0;
        const double total = r.sumProduct(Measure::iterations, Measure::elapsed);
        const std::string& name = r.config().mBenchmarkName;
        char rel[32];
        if (name.rfind("conv", 0) == 0) std::snprintf(rel, sizeof rel, "-");      // une conversion n'est pas comparable a la ref
        else                            std::snprintf(rel, sizeof rel, "%.1f%%", nsBase / ns * 100.0);
        std::printf("| %10s | %12.3f | %6.1f%% | %8.3f s | %s\n",
            rel, ns, err, total, name.c_str());
        if (pairs && i % 2 == 1 && i + 1 < res.size())
            std::printf("|            |              |         |            |\n");
    }
}

static std::string fmtMem(double bytes) {
    char buf[32];
    if (bytes >= 1048576.0) std::snprintf(buf, sizeof buf, "%.1f Mo", bytes / 1048576.0);
    else                    std::snprintf(buf, sizeof buf, "%.1f Ko", bytes / 1024.0);
    return buf;
}

// ============================================================================
// 1) Traitements par lots
// ============================================================================
struct Versions { bool novec, sse, soa, asmv; };

static void benchBatch(size_t n, const std::string& what, const Versions& v) {
    ankerl::nanobench::Rng rng(kSeed);

    // Donnees preparees hors chronometrage (les traitements ne modifient pas leurs entrees : pas de reinitialisation)
    std::vector<Vector3> a(n), b(n), outV(n);
    std::vector<float> outF(n);
    for (auto& x : a) x = randomVec(rng);
    for (auto& x : b) x = randomVec(rng);
    const Mat4x4f m = Mat4x4f::translate(1.5f, -2.0f, 3.0f) * Mat4x4f::rotationZ(0.7f);
    batch::Vec3SoA sa(n), sb(n), so(n);
    batch::aos_to_soa(a.data(), sa, n);
    batch::aos_to_soa(b.data(), sb, n);

    std::printf("\n=========== n = %zu ===========", n);
    const std::string ns = " (n = " + std::to_string(n) + ")";

    if (what == "all" || what == "dot") {
        Bench bench = makeBench(n, kWarmup, "element");
        bench.run("dot ref (AoS, autovec autorisee)", [&] { batch::dot_ref(a.data(), b.data(), outF.data(), n); doNotOptimizeAway(outF[n / 2]); });
        if (v.novec) bench.run("dot ref sans autovec", [&] { batch::dot_ref_novec(a.data(), b.data(), outF.data(), n); doNotOptimizeAway(outF[n / 2]); });
        if (v.sse)   bench.run("dot sse AoS", [&] { batch::dot_sse(a.data(), b.data(), outF.data(), n); doNotOptimizeAway(outF[n / 2]); });
        if (v.asmv)  bench.run("dot asm AoS (1 vec/iter)", [&] { dot_batch_asm(a.data(), b.data(), outF.data(), n); doNotOptimizeAway(outF[n / 2]); });
        if (v.soa) {
            bench.run("conv AoS->SoA (1 tableau)", [&] { batch::aos_to_soa(a.data(), sa, n); doNotOptimizeAway(sa.x[n / 2]); });
            bench.run("dot sse SoA calcul seul", [&] { batch::dot_soa_sse(sa, sb, outF.data(), n); doNotOptimizeAway(outF[n / 2]); });
            bench.run("dot sse SoA total (2 conv + calcul)", [&] {
                batch::aos_to_soa(a.data(), sa, n); batch::aos_to_soa(b.data(), sb, n);
                batch::dot_soa_sse(sa, sb, outF.data(), n); doNotOptimizeAway(outF[n / 2]); });
        }
        printTable("dot" + ns + ", memoire touchee ~ " + fmtMem(36.0 * n), bench, "element", false);
    }
    if (what == "all" || what == "normalize") {
        Bench bench = makeBench(n, kWarmup, "element");
        bench.run("normalize ref (AoS, autovec autorisee)", [&] { batch::normalize_ref(a.data(), outV.data(), n); doNotOptimizeAway(outV[n / 2].x); });
        if (v.novec) bench.run("normalize ref sans autovec", [&] { batch::normalize_ref_novec(a.data(), outV.data(), n); doNotOptimizeAway(outV[n / 2].x); });
        if (v.sse)   bench.run("normalize sse AoS", [&] { batch::normalize_sse(a.data(), outV.data(), n); doNotOptimizeAway(outV[n / 2].x); });
        if (v.soa) {
            bench.run("conv AoS->SoA", [&] { batch::aos_to_soa(a.data(), sa, n); doNotOptimizeAway(sa.x[n / 2]); });
            bench.run("normalize sse SoA calcul seul", [&] { batch::normalize_soa_sse(sa, so, n); doNotOptimizeAway(so.x[n / 2]); });
            bench.run("conv SoA->AoS", [&] { batch::soa_to_aos(so, outV.data(), n); doNotOptimizeAway(outV[n / 2].x); });
            bench.run("normalize sse SoA total (conv + calcul + conv)", [&] {
                batch::aos_to_soa(a.data(), sa, n); batch::normalize_soa_sse(sa, so, n);
                batch::soa_to_aos(so, outV.data(), n); doNotOptimizeAway(outV[n / 2].x); });
        }
        printTable("normalize" + ns + ", memoire touchee ~ " + fmtMem(32.0 * n), bench, "element", false);
    }
    if (what == "all" || what == "transform") {
        Bench bench = makeBench(n, kWarmup, "element");
        bench.run("transform ref (AoS, autovec autorisee)", [&] { batch::transform_ref(m, a.data(), outV.data(), n); doNotOptimizeAway(outV[n / 2].x); });
        if (v.novec) bench.run("transform ref sans autovec", [&] { batch::transform_ref_novec(m, a.data(), outV.data(), n); doNotOptimizeAway(outV[n / 2].x); });
        if (v.sse)   bench.run("transform sse AoS", [&] { batch::transform_sse(m, a.data(), outV.data(), n); doNotOptimizeAway(outV[n / 2].x); });
        if (v.soa) {
            bench.run("conv AoS->SoA", [&] { batch::aos_to_soa(a.data(), sa, n); doNotOptimizeAway(sa.x[n / 2]); });
            bench.run("transform sse SoA calcul seul", [&] { batch::transform_soa_sse(m, sa, so, n); doNotOptimizeAway(so.x[n / 2]); });
            bench.run("conv SoA->AoS", [&] { batch::soa_to_aos(so, outV.data(), n); doNotOptimizeAway(outV[n / 2].x); });
            bench.run("transform sse SoA total (conv + calcul + conv)", [&] {
                batch::aos_to_soa(a.data(), sa, n); batch::transform_soa_sse(m, sa, so, n);
                batch::soa_to_aos(so, outV.data(), n); doNotOptimizeAway(outV[n / 2].x); });
        }
        printTable("transform" + ns + ", memoire touchee ~ " + fmtMem(32.0 * n), bench, "element", false);
    }
}

// ============================================================================
// 2) Fonctions isolees (1 appel = 1 operation) : COMPLEMENT hors cahier.
//    Les temps incluent le surcout du harnais (index masque, doNotOptimizeAway) : les ratios sont sous-estimes.
// ============================================================================
static const size_t kMicroN = 256;
static const size_t kMask = kMicroN - 1;

static void benchVector3Functions() {
    ankerl::nanobench::Rng rng(kSeed);
    std::vector<ScalarVec3> sa(kMicroN), sb(kMicroN);
    std::vector<Vector3> va(kMicroN), vb(kMicroN);
    for (size_t j = 0; j < kMicroN; ++j) {
        float ax = randomFloat(rng), ay = randomFloat(rng), az = randomFloat(rng);
        float bx = randomFloat(rng), by = randomFloat(rng), bz = randomFloat(rng);
        sa[j] = ScalarVec3(ax, ay, az);  va[j] = Vector3(ax, ay, az);
        sb[j] = ScalarVec3(bx, by, bz);  vb[j] = Vector3(bx, by, bz);
    }
    size_t i = 0;
    Bench dotBench = makeBench(1, 100, "op");
    dotBench.run("dot scalaire", [&] { size_t k = i++ & kMask; doNotOptimizeAway(sa[k].dot(sb[k])); });
    dotBench.run("dot simd", [&] { size_t k = i++ & kMask; doNotOptimizeAway(va[k].dot(vb[k])); });
    dotBench.run("dot asm", [&] { size_t k = i++ & kMask; doNotOptimizeAway(dot_vec3_asm(&va[k].x, &vb[k].x)); });
    printTable("Vector3::dot (1 appel, complement hors cahier, surcout du harnais inclus)", dotBench, "op", false);

    Bench bench = makeBench(1, 100, "op");
    bench.run("magnitude scalaire", [&] { size_t k = i++ & kMask; doNotOptimizeAway(sa[k].magnitude()); });
    bench.run("magnitude simd", [&] { size_t k = i++ & kMask; doNotOptimizeAway(va[k].magnitude()); });
    bench.run("normalized scalaire", [&] { size_t k = i++ & kMask; doNotOptimizeAway(sa[k].normalized()); });
    bench.run("normalized simd", [&] { size_t k = i++ & kMask; doNotOptimizeAway(va[k].normalized()); });
    printTable("Vector3::magnitude / normalized (1 appel, complement hors cahier)", bench, "op", true);
}

static void benchMatrixFunctions() {
    ankerl::nanobench::Rng rng(kSeed);
    std::vector<float> t(kMicroN), ang(kMicroN);
    for (size_t j = 0; j < kMicroN; ++j) { t[j] = randomFloat(rng); ang[j] = float(rng.uniform01()) * 6.2831853f; }
    std::vector<ScalarMat4> sm(kMicroN), sn(kMicroN);
    std::vector<Mat4x4f> vm(kMicroN), vn(kMicroN);
    for (size_t j = 0; j < kMicroN; ++j) {
        const float tx = t[j], ty = t[(j + 1) & kMask], tz = t[(j + 2) & kMask];
        const float ux = t[(j + 3) & kMask], uy = t[(j + 4) & kMask], uz = t[(j + 5) & kMask];
        sm[j] = ScalarMat4::translate(tx, ty, tz) * ScalarMat4::rotationZ(ang[j]);
        sn[j] = ScalarMat4::rotationZ(ang[(j + 7) & kMask]) * ScalarMat4::translate(ux, uy, uz);
        vm[j] = Mat4x4f::translate(tx, ty, tz) * Mat4x4f::rotationZ(ang[j]);
        vn[j] = Mat4x4f::rotationZ(ang[(j + 7) & kMask]) * Mat4x4f::translate(ux, uy, uz);
    }
    size_t i = 0;
    Bench bench = makeBench(1, 100, "op");

    bench.run("translate scalaire", [&] { size_t k = i++ & kMask; doNotOptimizeAway(ScalarMat4::translate(t[k], t[(k + 1) & kMask], t[(k + 2) & kMask])); });
    bench.run("translate simd", [&] { size_t k = i++ & kMask; doNotOptimizeAway(Mat4x4f::translate(t[k], t[(k + 1) & kMask], t[(k + 2) & kMask])); });
    bench.run("rotationZ scalaire", [&] { size_t k = i++ & kMask; doNotOptimizeAway(ScalarMat4::rotationZ(ang[k])); });
    bench.run("rotationZ simd", [&] { size_t k = i++ & kMask; doNotOptimizeAway(Mat4x4f::rotationZ(ang[k])); });
    bench.run("operator* scalaire", [&] { size_t k = i++ & kMask; doNotOptimizeAway(sm[k] * sn[k]); });
    bench.run("operator* simd", [&] { size_t k = i++ & kMask; doNotOptimizeAway(vm[k] * vn[k]); });

    printTable("Mat4x4 (1 appel, complement hors cahier, surcout du harnais inclus)", bench, "op", true);
}

// ============================================================================
static bool isNumber(const char* s) { return s && *s && std::strspn(s, "0123456789") == std::strlen(s); }

int main(int argc, char** argv) {
    std::string what = (argc > 1) ? argv[1] : "all";
    std::string ver = "all";
    int firstSize = 2;
    if (argc > 2 && !isNumber(argv[2])) { ver = argv[2]; firstSize = 3; }

    std::vector<size_t> sizes = { 16, 4096, 1048576 };     // petit, moyen, grand
    if (argc > firstSize) {
        sizes.clear();
        for (int i = firstSize; i < argc; ++i) {
            if (!isNumber(argv[i])) { std::printf("Taille invalide : %s\n", argv[i]); return 1; }
            sizes.push_back(std::stoull(argv[i]));
        }
    }

    const bool isBatch = (what == "all" || what == "dot" || what == "normalize" || what == "transform");
    const bool isMicro = (what == "all" || what == "vector3" || what == "matrix");
    const bool validVer = (ver == "all" || ver == "novec" || ver == "sse" || ver == "soa" || ver == "asm");
    if ((!isBatch && !isMicro && what != "check") || !validVer) {
        std::printf("Usage : Benchmark.exe [all|dot|normalize|transform|vector3|matrix|check] [all|novec|sse|soa|asm] [tailles...]\n");
        return 1;
    }
    if (ver == "asm" && what != "all" && what != "dot") {
        std::printf("La version asm n'existe que pour dot.\n");
        return 1;
    }
    const Versions v{ ver == "all" || ver == "novec", ver == "all" || ver == "sse",
                      ver == "all" || ver == "soa",   ver == "all" || ver == "asm" };

    std::printf("Benchmark | Release x64, sans debogueur (Ctrl+F5) | seed = %u | warmup = %zu | epoques = %zu\n", kSeed, kWarmup, kEpochs);
    printEnv();
    std::printf("Relancer avec :");
    for (int i = 0; i < argc; ++i) std::printf(" %s", i == 0 ? "Benchmark.exe" : argv[i]);
    std::printf("\n");
    std::printf("relative = temps ligne de base / temps ligne (100%% = base, 300%% = 3x plus rapide) | err%% = erreur absolue mediane en %% (dispersion)\n");
    std::printf("Base = reference C++ optimisee (autovec autorisee). \"sans autovec\" = variante separee (#pragma loop(no_vector)).\n");

    pinToPerformanceCore();
    if (!runSelfChecks()) return 2;                         // pas de mesure si un resultat est faux
    if (what == "check") return 0;

    if (isBatch) {
        std::printf("\ntailles :");
        for (size_t n : sizes) std::printf(" %zu", n);
        std::printf("\n");
        for (size_t n : sizes) {
            if (n == 0) continue;
            benchBatch(n, what, v);
        }
    }
    if (what == "all" || what == "vector3") benchVector3Functions();
    if (what == "all" || what == "matrix")  benchMatrixFunctions();
    return 0;
}