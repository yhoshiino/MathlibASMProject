#pragma once
#include <cstddef>
#include "Vector3SIMD.h"   // math::simd::Vector3 (16 octets, aligné 16, w = padding)
#include "Matrix4x4SIMD.h"   // math::simd::Mat4x4f (column-major, vecteurs colonnes)

// ============================================================================
// Traitements par lots, layout AoS : tableaux de Vector3 (x, y, z, w=padding).
//
// Chaque traitement existe en 2 versions :
//   *_ref : C++ scalaire, formules écrites à la main (aucun intrinsic).
//           Reste disponible pour les tests et les comparaisons.
//   *_sse : SIMD explicite, intrinsics SSE/SSE2 uniquement.
//
// Contrats communs :
//   - n peut valoir 0 (rien n'est lu ni écrit), 1, 3, 4, 5, ... (aucune contrainte de multiple de 4).
//   - Aucune lecture/écriture au-delà de [0, n).
//   - Alignement : les Vector3* doivent être alignés sur 16 octets (garanti par le type :
//     std::vector<Vector3> et les tableaux le respectent). Le float* de dot_* n'a aucune
//     contrainte d'alignement (stores non alignés).
//   - Les tableaux d'entrée et de sortie ne doivent pas se chevaucher partiellement.
//   - Les deux versions suivent exactement les mêmes conventions et le même ordre d'addition.
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

} // namespace math::batch