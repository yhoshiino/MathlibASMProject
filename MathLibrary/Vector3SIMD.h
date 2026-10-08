#pragma once
#include <emmintrin.h> // SSE2 uniquement

namespace math::simd {

    // ------------------------------------------------------------------
    // Convention de la bibliothèque (à reprendre dans la documentation) :
    //  - Vector3 occupe 16 octets : x, y, z, w (w = padding, toujours 0 en pratique,
    //    jamais lu par les calculs : dot/normalized/magnitude l'ignorent).
    //  - Alignement 16 octets (alignas(16)) : load/store alignés possibles en AoS.
    //  - Vecteur nul : normalized() renvoie (0,0,0) si |v|^2 <= kNormalizeEpsSq.
    //    Les versions de référence ET SIMD utilisent la même valeur.
    // ------------------------------------------------------------------

    // Seuil unique et documenté pour le vecteur nul (|v| < 1e-6).
    inline constexpr float kNormalizeEpsSq = 1e-12f;

    namespace detail {
        // Produit scalaire SSE2 sur x,y,z (w ignoré), résultat dans les 4 lanes.
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

        // --- Produit scalaire & magnitude (SSE2 pur : mul + shuffle + add) ---
        float dot(const Vector3& o) const {
            return _mm_cvtss_f32(detail::dot_xyz(reg, o.reg));
        }

        float magnitude() const {
            return _mm_cvtss_f32(_mm_sqrt_ss(detail::dot_xyz(reg, reg)));
        }

        // sqrt + div exacts (pas de rsqrt approximatif),
        // vecteur nul géré sans branche avec le même seuil que la référence.
        Vector3 normalized() const {
            __m128 len2 = detail::dot_xyz(reg, reg);
            __m128 ok = _mm_cmpgt_ps(len2, _mm_set1_ps(kNormalizeEpsSq)); // faux si NaN aussi
            __m128 res = _mm_div_ps(reg, _mm_sqrt_ps(len2));
            return _mm_and_ps(res, _mm_and_ps(ok, detail::mask_xyz()));   // w forcé à 0
        }
    };

} // namespace math::simd