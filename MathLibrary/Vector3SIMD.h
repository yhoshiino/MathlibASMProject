#pragma once
#include <cmath>
#include <iostream>
#include <string>
#include <algorithm>
#include <limits>
#include <array>
#include <stdexcept>
#include <emmintrin.h> // [CHANGÉ] SSE2 uniquement (plus de smmintrin.h / _mm_dp_ps)

namespace math::simd {

    // ------------------------------------------------------------------
    // Convention de la bibliothèque (à reprendre dans la documentation) :
    //  - Vector3 occupe 16 octets : x, y, z, w (w = padding, toujours 0 en pratique,
    //    jamais lu par les calculs : dot/normalized/magnitude l'ignorent).
    //  - Alignement 16 octets (alignas(16)) : load/store alignés possibles en AoS.
    //  - Vecteur nul : normalized() renvoie (0,0,0) si |v|^2 <= kNormalizeEpsSq.
    //    Les versions de référence ET SIMD doivent utiliser cette même constante.
    // ------------------------------------------------------------------

    // [CHANGÉ] Seuil unique et documenté pour le vecteur nul (|v| < 1e-6).
    inline constexpr float kNormalizeEpsSq = 1e-12f;

    namespace detail {
        // [CHANGÉ] Produit scalaire SSE2 sur x,y,z (w ignoré), résultat dans les 4 lanes.
        // Ordre d'addition : (x + y) + z, identique à la version scalaire x*x + y*y + z*z.
        inline __m128 dot_xyz(__m128 a, __m128 b) {
            __m128 m = _mm_mul_ps(a, b);                                   // x y z w
            __m128 y = _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 1, 1, 1));
            __m128 z = _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 2, 2, 2));
            __m128 s = _mm_add_ss(_mm_add_ss(m, y), z);                    // lane 0 = x+y+z
            return _mm_shuffle_ps(s, s, _MM_SHUFFLE(0, 0, 0, 0));          // broadcast
        }

        inline __m128 mask_xyz() {
            return _mm_castsi128_ps(_mm_setr_epi32(-1, -1, -1, 0));
        }
    }

    class alignas(16) Vector3 {
    public:
        // NB : struct anonyme dans une union = extension MSVC (OK pour ce projet, à mentionner).
        union {
            __m128 reg;
            struct { float x, y, z, w; };
        };

        // --- Constructeurs ---
        Vector3() : reg(_mm_setzero_ps()) {}
        Vector3(float x, float y, float z) : reg(_mm_setr_ps(x, y, z, 0.0f)) {}
        Vector3(__m128 m) : reg(m) {}

        // --- Factory / Constantes ---
        static Vector3 up() { return Vector3(0.0f, 1.0f, 0.0f); }
        static Vector3 down() { return Vector3(0.0f, -1.0f, 0.0f); }
        static Vector3 left() { return Vector3(-1.0f, 0.0f, 0.0f); }
        static Vector3 right() { return Vector3(1.0f, 0.0f, 0.0f); }
        static Vector3 forward() { return Vector3(0.0f, 0.0f, 1.0f); }
        static Vector3 back() { return Vector3(0.0f, 0.0f, -1.0f); }
        static Vector3 one() { return Vector3(1.0f, 1.0f, 1.0f); }
        static Vector3 zero() { return Vector3(_mm_setzero_ps()); }
        static Vector3 negativeInfinity() {
            const float inf = std::numeric_limits<float>::infinity();
            return Vector3(-inf, -inf, -inf);
        }
        static Vector3 positiveInfinity() {
            const float inf = std::numeric_limits<float>::infinity();
            return Vector3(inf, inf, inf);
        }

        // --- Opérateurs SIMD de base ---
        Vector3 operator+(const Vector3& o) const { return _mm_add_ps(reg, o.reg); }
        Vector3 operator-(const Vector3& o) const { return _mm_sub_ps(reg, o.reg); }
        Vector3 operator*(float scalar)     const { return _mm_mul_ps(reg, _mm_set1_ps(scalar)); }
        Vector3 operator/(float scalar)     const { return _mm_div_ps(reg, _mm_set1_ps(scalar)); }

        Vector3& operator+=(const Vector3& o) { reg = _mm_add_ps(reg, o.reg); return *this; }
        Vector3& operator-=(const Vector3& o) { reg = _mm_sub_ps(reg, o.reg); return *this; }
        Vector3& operator*=(float scalar) { reg = _mm_mul_ps(reg, _mm_set1_ps(scalar)); return *this; }
        Vector3& operator/=(float scalar) { reg = _mm_div_ps(reg, _mm_set1_ps(scalar)); return *this; }

        // --- Produit scalaire, vectoriel & Magnitudes ---
        // [CHANGÉ] SSE2 pur (mul + shuffle + add) au lieu de _mm_dp_ps (SSE4.1)
        float dot(const Vector3& o) const {
            return _mm_cvtss_f32(detail::dot_xyz(reg, o.reg));
        }

        Vector3 cross(const Vector3& o) const {
            __m128 a_yzx = _mm_shuffle_ps(reg, reg, _MM_SHUFFLE(3, 0, 2, 1));
            __m128 b_yzx = _mm_shuffle_ps(o.reg, o.reg, _MM_SHUFFLE(3, 0, 2, 1));
            __m128 a_zxy = _mm_shuffle_ps(reg, reg, _MM_SHUFFLE(3, 1, 0, 2));
            __m128 b_zxy = _mm_shuffle_ps(o.reg, o.reg, _MM_SHUFFLE(3, 1, 0, 2));
            return _mm_sub_ps(_mm_mul_ps(a_yzx, b_zxy), _mm_mul_ps(a_zxy, b_yzx));
        }

        float sqrMagnitude() const {
            return dot(*this);
        }

        float magnitude() const {
            return _mm_cvtss_f32(_mm_sqrt_ss(detail::dot_xyz(reg, reg)));
        }

        // [CHANGÉ] sqrt + div exacts (plus de _mm_rsqrt_ps approximatif),
        // vecteur nul géré sans branche avec le même seuil que la référence.
        Vector3 normalized() const {
            __m128 len2 = detail::dot_xyz(reg, reg);
            __m128 ok = _mm_cmpgt_ps(len2, _mm_set1_ps(kNormalizeEpsSq)); // faux si NaN aussi
            __m128 res = _mm_div_ps(reg, _mm_sqrt_ps(len2));
            return _mm_and_ps(res, _mm_and_ps(ok, detail::mask_xyz()));   // w forcé à 0
        }

        // --- Méthodes Statiques Géométriques ---
        static float distance(const Vector3& a, const Vector3& b) {
            return (a - b).magnitude();
        }

        static Vector3 Lerp(const Vector3& a, const Vector3& b, float t) {
            t = std::clamp(t, 0.0f, 1.0f); // [CHANGÉ] std::clamp (C++17/20)
            return LerpUnclamped(a, b, t);
        }

        static Vector3 LerpUnclamped(const Vector3& a, const Vector3& b, float t) {
            __m128 vt = _mm_set1_ps(t);
            return _mm_add_ps(a.reg, _mm_mul_ps(_mm_sub_ps(b.reg, a.reg), vt));
        }

        static Vector3 Max(const Vector3& a, const Vector3& b) { return _mm_max_ps(a.reg, b.reg); }
        static Vector3 Min(const Vector3& a, const Vector3& b) { return _mm_min_ps(a.reg, b.reg); }

        static Vector3 MoveTowards(const Vector3& current, const Vector3& target, float maxDelta) {
            Vector3 delta = target - current;
            float sqrDist = delta.sqrMagnitude();
            if (sqrDist <= maxDelta * maxDelta || sqrDist == 0.0f) return target;
            float dist = std::sqrt(sqrDist);
            return current + delta * (maxDelta / dist);
        }

        // --- Transformées & Utilitaires ---
        Vector3 Reflect(const Vector3& normal) const {
            return *this - normal * (2.0f * this->dot(normal));
        }

        Vector3 Scale(const Vector3& other) const {
            return _mm_mul_ps(reg, other.reg);
        }

        Vector3 ClampMagnitude(float maxVal) const {
            float sqrMag = sqrMagnitude();
            if (sqrMag > maxVal * maxVal) {
                float scale = maxVal / std::sqrt(sqrMag);
                return *this * scale;
            }
            return *this;
        }

        void SetVector3(float newX, float newY, float newZ) {
            reg = _mm_setr_ps(newX, newY, newZ, 0.0f);
        }

        // --- Accès et Comparaisons ---
        float operator[](int index) const {
            if (index == 0) return x;
            if (index == 1) return y;
            if (index == 2) return z;
            throw std::out_of_range("Index out of range");
        }

        bool operator==(const Vector3& o) const {
            __m128 cmp = _mm_cmpeq_ps(reg, o.reg);
            return (_mm_movemask_ps(cmp) & 0x7) == 0x7; // x, y, z uniquement
        }

        bool operator!=(const Vector3& o) const { return !(*this == o); }

        std::string toString() const {
            return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
        }

        void print() const {
            std::cout << "(" << x << ", " << y << ", " << z << ")\n";
        }

        std::array<float, 3> toArray() const { return { x, y, z }; }
    };
}