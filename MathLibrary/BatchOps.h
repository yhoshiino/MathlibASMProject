#pragma once
#include <cstddef>
#include <vector>
#include "Vector3SIMD.h"   // math::simd::Vector3 (16 octets, aligné 16, w = padding)
#include "Matrix4x4SIMD.h"   // math::simd::Mat4x4f (column-major, vecteurs colonnes)

// ============================================================================
// Traitements par lots, layout AoS : tableaux de Vector3 (x, y, z, w=padding).
//
// Chaque traitement existe en plusieurs versions :
//   *_ref       : C++ scalaire, formules écrites à la main (aucun intrinsic). Référence optimisée
//                 (auto-vectorisation autorisée), reste disponible pour les tests et les comparaisons.
//   *_ref_novec : même code, auto-vectorisation désactivée (#pragma loop(no_vector)). À présenter séparément.
//   *_sse       : SIMD explicite, intrinsics SSE/SSE2 uniquement (AoS : 1 registre = 1 vecteur).
//   *_soa_sse   : SIMD explicite en SoA (1 registre = même composante de 4 vecteurs).
//   dot_*_asm   : ASM x64 (DotVec3.asm).
//
// Contrats communs :
//   - n peut valoir 0 (rien n'est lu ni écrit), 1, 3, 4, 5, ... (aucune contrainte de multiple de 4).
//   - Aucune lecture/écriture au-delà de [0, n).
//   - Alignement : les Vector3* doivent être alignés sur 16 octets (garanti par le type :
//     std::vector<Vector3> et les tableaux le respectent). Le float* de dot_* n'a aucune
//     contrainte d'alignement (stores non alignés). Les versions SoA utilisent loadu/storeu :
//     aucune contrainte d'alignement.
//   - Les tableaux d'entrée et de sortie ne doivent pas se chevaucher partiellement.
//   - Toutes les versions suivent les mêmes conventions et le même ordre d'addition : leurs résultats
//     sont identiques bit à bit (vérifié au lancement de Benchmark.exe).
// ============================================================================

namespace math::batch {

    // ---- 1. Produits scalaires : out[i] = a[i] . b[i]  =  (x*x' + y*y') + z*z'  (w ignoré)
    void dot_ref(const simd::Vector3* a, const simd::Vector3* b, float* out, std::size_t n);
    void dot_sse(const simd::Vector3* a, const simd::Vector3* b, float* out, std::size_t n);

    // ---- 2. Normalisation : out[i] = in[i] / |in[i]|
    //      Vecteur nul : si |v|^2 <= simd::kNormalizeEpsSq (1e-12) ou NaN, le résultat est (0,0,0).
    //      Résultat : w = 0.
    void normalize_ref(const simd::Vector3* in, simd::Vector3* out, std::size_t n);
    void normalize_sse(const simd::Vector3* in, simd::Vector3* out, std::size_t n);

    // ---- 3. Transformation de points par une matrice AFFINE 4x4 (dernière ligne = 0,0,0,1) :
    //      out[i] = M * (x, y, z, 1)   (w = 1, aucune division perspective). Résultat : w = 0.
    //      x' = ((m00*x + m01*y) + m02*z) + m03, etc.
    void transform_ref(const simd::Mat4x4f& m, const simd::Vector3* in, simd::Vector3* out, std::size_t n);
    void transform_sse(const simd::Mat4x4f& m, const simd::Vector3* in, simd::Vector3* out, std::size_t n);

    // ---- 4. SoA (structure de tableaux) : trois tableaux x[], y[], z[].
    //      Un registre SSE contient la MÊME composante de 4 vecteurs différents : aucun shuffle, aucune lane w gâchée.
    //      Les tableaux SoA doivent contenir au moins n éléments ; les conversions n'allouent rien.
    //      SoA -> AoS : w = 0.
    struct Vec3SoA {
        std::vector<float> x, y, z;
        Vec3SoA() = default;
        explicit Vec3SoA(std::size_t n) : x(n), y(n), z(n) {}
    };

    // Conversions (SSE : 4 vecteurs -> transposition 4x4). SoA -> AoS : w = 0.
    void aos_to_soa(const simd::Vector3* in, Vec3SoA& out, std::size_t n);
    void soa_to_aos(const Vec3SoA& in, simd::Vector3* out, std::size_t n);

    // Traitements SoA (mêmes formules, même ordre d'opérations que les versions AoS)
    void dot_soa_sse(const Vec3SoA& a, const Vec3SoA& b, float* out, std::size_t n);
    void normalize_soa_sse(const Vec3SoA& in, Vec3SoA& out, std::size_t n);
    void transform_soa_sse(const simd::Mat4x4f& m, const Vec3SoA& in, Vec3SoA& out, std::size_t n);

    // Références sans auto-vectorisation (corps identiques à *_ref)
    void dot_ref_novec(const simd::Vector3* a, const simd::Vector3* b, float* out, std::size_t n);
    void normalize_ref_novec(const simd::Vector3* in, simd::Vector3* out, std::size_t n);
    void transform_ref_novec(const simd::Mat4x4f& m, const simd::Vector3* in, simd::Vector3* out, std::size_t n);

} // namespace math::batch

// ---- ASM x64 (DotVec3.asm). Convention Windows x64 : args rcx, rdx, r8, r9 ; float retourné dans xmm0.
extern "C" {
    // dot de UN couple de Vec3 (pointeurs vers x,y,z,[w]) : lit 16 octets par pointeur (x y z w).
    float dot_vec3_asm(const float* a, const float* b);
    // dot par lots, 1 vecteur par itération : out[i] = a[i] . b[i]. n = 0 autorisé.
    void  dot_batch_asm(const math::simd::Vector3* a, const math::simd::Vector3* b, float* out, std::size_t n);
}