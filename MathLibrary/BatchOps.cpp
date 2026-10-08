#include "BatchOps.h"
#include <cmath>
#include <emmintrin.h>

namespace math::batch {

    using simd::Vector3;
    using simd::Mat4x4f;

    // ========================================================================
    // 1. DOT
    // ========================================================================

    void dot_ref(const Vector3* a, const Vector3* b, float* out, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = a[i].x * b[i].x + a[i].y * b[i].y + a[i].z * b[i].z;
    }

    // Layout AoS : un registre = UN vecteur (x y z w).
    // On traite 4 vecteurs à la fois : 4 multiplications, puis une transposition 4x4
    // pour obtenir  r0 = [x0 x1 x2 x3], r1 = [y...], r2 = [z...]  -> (r0 + r1) + r2 = 4 dots.
    void dot_sse(const Vector3* a, const Vector3* b, float* out, std::size_t n)
    {
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 m0 = _mm_mul_ps(a[i + 0].reg, b[i + 0].reg);
            __m128 m1 = _mm_mul_ps(a[i + 1].reg, b[i + 1].reg);
            __m128 m2 = _mm_mul_ps(a[i + 2].reg, b[i + 2].reg);
            __m128 m3 = _mm_mul_ps(a[i + 3].reg, b[i + 3].reg);

            _MM_TRANSPOSE4_PS(m0, m1, m2, m3);   // m0 = xs, m1 = ys, m2 = zs, m3 = ws (ignoré)

            __m128 r = _mm_add_ps(_mm_add_ps(m0, m1), m2);
            _mm_storeu_ps(out + i, r);
        }
        // Reste (n % 4 éléments, ou n == 0) : même formule, même ordre d'addition.
        for (; i < n; ++i)
            out[i] = a[i].x * b[i].x + a[i].y * b[i].y + a[i].z * b[i].z;
    }

    // ========================================================================
    // 2. NORMALIZE
    // ========================================================================

    void normalize_ref(const Vector3* in, Vector3* out, std::size_t n)
    {
        for (std::size_t i = 0; i < n; ++i) {
            const float x = in[i].x, y = in[i].y, z = in[i].z;
            const float len2 = x * x + y * y + z * z;
            if (!(len2 > simd::kNormalizeEpsSq)) {       // vecteur nul (ou NaN)
                out[i] = Vector3(0.0f, 0.0f, 0.0f);
            }
            else {
                const float len = std::sqrt(len2);
                out[i] = Vector3(x / len, y / len, z / len);
            }
        }
    }

    // Layout AoS : un registre = UN vecteur. Chaque itération traite 1 vecteur (4 lanes = x y z w),
    // donc aucun "reste" à gérer : n quelconque, y compris 0.
    void normalize_sse(const Vector3* in, Vector3* out, std::size_t n)
    {
        const __m128 eps = _mm_set1_ps(simd::kNormalizeEpsSq);
        const __m128 mask = simd::detail::mask_xyz();

        for (std::size_t i = 0; i < n; ++i) {
            const __m128 v = in[i].reg;

            // len2 dans les 4 lanes : (x*x + y*y) + z*z
            const __m128 sq = _mm_mul_ps(v, v);
            const __m128 sy = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(1, 1, 1, 1));
            const __m128 sz = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(2, 2, 2, 2));
            __m128 len2 = _mm_add_ss(_mm_add_ss(sq, sy), sz);
            len2 = _mm_shuffle_ps(len2, len2, _MM_SHUFFLE(0, 0, 0, 0));

            const __m128 ok = _mm_cmpgt_ps(len2, eps);              // faux si NaN
            const __m128 res = _mm_div_ps(v, _mm_sqrt_ps(len2));     // exact (pas de rsqrt)
            out[i].reg = _mm_and_ps(res, _mm_and_ps(ok, mask));      // nul -> 0, et w = 0
        }
    }

    // ========================================================================
    // 3. TRANSFORM (matrice affine, w = 1, sans division perspective)
    // ========================================================================

    void transform_ref(const Mat4x4f& m, const Vector3* in, Vector3* out, std::size_t n)
    {
        // column-major : élément (row, col) = d[col * 4 + row]
        const float* d = m.data;
        const float m00 = d[0], m10 = d[1], m20 = d[2];
        const float m01 = d[4], m11 = d[5], m21 = d[6];
        const float m02 = d[8], m12 = d[9], m22 = d[10];
        const float m03 = d[12], m13 = d[13], m23 = d[14];

        for (std::size_t i = 0; i < n; ++i) {
            const float x = in[i].x, y = in[i].y, z = in[i].z;
            out[i] = Vector3(m00 * x + m01 * y + m02 * z + m03,
                m10 * x + m11 * y + m12 * z + m13,
                m20 * x + m21 * y + m22 * z + m23);
        }
    }

    // Layout AoS : un registre = UN point. Les 4 lanes du résultat = (x', y', z', w') du même point :
    //   r = col0 * x + col1 * y + col2 * z + col3     (w = 1 implicite)
    // Un point par itération : aucun reste.
    void transform_sse(const Mat4x4f& m, const Vector3* in, Vector3* out, std::size_t n)
    {
        const __m128 c0 = m.col[0], c1 = m.col[1], c2 = m.col[2], c3 = m.col[3];
        const __m128 mask = simd::detail::mask_xyz();

        for (std::size_t i = 0; i < n; ++i) {
            const __m128 p = in[i].reg;
            const __m128 x = _mm_shuffle_ps(p, p, _MM_SHUFFLE(0, 0, 0, 0));
            const __m128 y = _mm_shuffle_ps(p, p, _MM_SHUFFLE(1, 1, 1, 1));
            const __m128 z = _mm_shuffle_ps(p, p, _MM_SHUFFLE(2, 2, 2, 2));

            __m128 r = _mm_mul_ps(c0, x);
            r = _mm_add_ps(r, _mm_mul_ps(c1, y));
            r = _mm_add_ps(r, _mm_mul_ps(c2, z));
            r = _mm_add_ps(r, c3);

            out[i].reg = _mm_and_ps(r, mask);   // w = 0 (convention Vector3)
        }
    }

} // namespace math::batch