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

    // ========================================================================
    // Conversions AoS <-> SoA
    // ========================================================================

    void aos_to_soa(const Vector3* in, Vec3SoA& out, std::size_t n)
    {
        float* ox = out.x.data(); float* oy = out.y.data(); float* oz = out.z.data();
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 r0 = in[i + 0].reg, r1 = in[i + 1].reg, r2 = in[i + 2].reg, r3 = in[i + 3].reg;
            _MM_TRANSPOSE4_PS(r0, r1, r2, r3);          // r0 = xs, r1 = ys, r2 = zs, r3 = ws (jeté)
            _mm_storeu_ps(ox + i, r0);
            _mm_storeu_ps(oy + i, r1);
            _mm_storeu_ps(oz + i, r2);
        }
        for (; i < n; ++i) { ox[i] = in[i].x; oy[i] = in[i].y; oz[i] = in[i].z; }
    }

    void soa_to_aos(const Vec3SoA& in, Vector3* out, std::size_t n)
    {
        const float* ix = in.x.data(); const float* iy = in.y.data(); const float* iz = in.z.data();
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 r0 = _mm_loadu_ps(ix + i), r1 = _mm_loadu_ps(iy + i), r2 = _mm_loadu_ps(iz + i);
            __m128 r3 = _mm_setzero_ps();               // w = 0
            _MM_TRANSPOSE4_PS(r0, r1, r2, r3);          // r0 = (x0 y0 z0 0), ...
            out[i + 0].reg = r0; out[i + 1].reg = r1; out[i + 2].reg = r2; out[i + 3].reg = r3;
        }
        for (; i < n; ++i) out[i] = Vector3(ix[i], iy[i], iz[i]);
    }

    // ========================================================================
    // Traitements SoA : lanes = 4 vecteurs différents, 0 shuffle
    // ========================================================================

    void dot_soa_sse(const Vec3SoA& a, const Vec3SoA& b, float* out, std::size_t n)
    {
        const float* ax = a.x.data(); const float* ay = a.y.data(); const float* az = a.z.data();
        const float* bx = b.x.data(); const float* by = b.y.data(); const float* bz = b.z.data();
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m128 r = _mm_mul_ps(_mm_loadu_ps(ax + i), _mm_loadu_ps(bx + i));
            r = _mm_add_ps(r, _mm_mul_ps(_mm_loadu_ps(ay + i), _mm_loadu_ps(by + i)));
            r = _mm_add_ps(r, _mm_mul_ps(_mm_loadu_ps(az + i), _mm_loadu_ps(bz + i)));   // (x+y)+z
            _mm_storeu_ps(out + i, r);
        }
        for (; i < n; ++i) out[i] = ax[i] * bx[i] + ay[i] * by[i] + az[i] * bz[i];
    }

    void normalize_soa_sse(const Vec3SoA& in, Vec3SoA& out, std::size_t n)
    {
        const float* ix = in.x.data(); const float* iy = in.y.data(); const float* iz = in.z.data();
        float* ox = out.x.data(); float* oy = out.y.data(); float* oz = out.z.data();
        const __m128 eps = _mm_set1_ps(simd::kNormalizeEpsSq);
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            const __m128 x = _mm_loadu_ps(ix + i), y = _mm_loadu_ps(iy + i), z = _mm_loadu_ps(iz + i);
            __m128 len2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y)), _mm_mul_ps(z, z));
            const __m128 ok = _mm_cmpgt_ps(len2, eps);          // faux si NaN ou nul
            const __m128 len = _mm_sqrt_ps(len2);               // 1 sqrtps pour 4 vecteurs
            _mm_storeu_ps(ox + i, _mm_and_ps(_mm_div_ps(x, len), ok));   // 1 divps par composante
            _mm_storeu_ps(oy + i, _mm_and_ps(_mm_div_ps(y, len), ok));
            _mm_storeu_ps(oz + i, _mm_and_ps(_mm_div_ps(z, len), ok));
        }
        for (; i < n; ++i) {
            const float x = ix[i], y = iy[i], z = iz[i];
            const float len2 = x * x + y * y + z * z;
            if (!(len2 > simd::kNormalizeEpsSq)) { ox[i] = 0.0f; oy[i] = 0.0f; oz[i] = 0.0f; }
            else { const float len = std::sqrt(len2); ox[i] = x / len; oy[i] = y / len; oz[i] = z / len; }
        }
    }

    void transform_soa_sse(const Mat4x4f& m, const Vec3SoA& in, Vec3SoA& out, std::size_t n)
    {
        const float* d = m.data;                                // column-major : (row, col) = d[col*4 + row]
        const float m00 = d[0], m10 = d[1], m20 = d[2];
        const float m01 = d[4], m11 = d[5], m21 = d[6];
        const float m02 = d[8], m12 = d[9], m22 = d[10];
        const float m03 = d[12], m13 = d[13], m23 = d[14];
        const __m128 c00 = _mm_set1_ps(m00), c01 = _mm_set1_ps(m01), c02 = _mm_set1_ps(m02), c03 = _mm_set1_ps(m03);
        const __m128 c10 = _mm_set1_ps(m10), c11 = _mm_set1_ps(m11), c12 = _mm_set1_ps(m12), c13 = _mm_set1_ps(m13);
        const __m128 c20 = _mm_set1_ps(m20), c21 = _mm_set1_ps(m21), c22 = _mm_set1_ps(m22), c23 = _mm_set1_ps(m23);

        const float* ix = in.x.data(); const float* iy = in.y.data(); const float* iz = in.z.data();
        float* ox = out.x.data(); float* oy = out.y.data(); float* oz = out.z.data();
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            const __m128 x = _mm_loadu_ps(ix + i), y = _mm_loadu_ps(iy + i), z = _mm_loadu_ps(iz + i);
            __m128 rx = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(c00, x), _mm_mul_ps(c01, y)), _mm_mul_ps(c02, z)), c03);
            __m128 ry = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(c10, x), _mm_mul_ps(c11, y)), _mm_mul_ps(c12, z)), c13);
            __m128 rz = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(c20, x), _mm_mul_ps(c21, y)), _mm_mul_ps(c22, z)), c23);
            _mm_storeu_ps(ox + i, rx); _mm_storeu_ps(oy + i, ry); _mm_storeu_ps(oz + i, rz);
        }
        for (; i < n; ++i) {
            const float x = ix[i], y = iy[i], z = iz[i];
            ox[i] = m00 * x + m01 * y + m02 * z + m03;
            oy[i] = m10 * x + m11 * y + m12 * z + m13;
            oz[i] = m20 * x + m21 * y + m22 * z + m23;
        }
    }

    // ========================================================================
    // Références SANS auto-vectorisation (variante identifiée séparément)
    // ========================================================================

    void dot_ref_novec(const Vector3* a, const Vector3* b, float* out, std::size_t n)
    {
#pragma loop(no_vector)
        for (std::size_t i = 0; i < n; ++i)
            out[i] = a[i].x * b[i].x + a[i].y * b[i].y + a[i].z * b[i].z;
    }

    void normalize_ref_novec(const Vector3* in, Vector3* out, std::size_t n)
    {
#pragma loop(no_vector)
        for (std::size_t i = 0; i < n; ++i) {
            const float x = in[i].x, y = in[i].y, z = in[i].z;
            const float len2 = x * x + y * y + z * z;
            if (!(len2 > simd::kNormalizeEpsSq)) {
                out[i] = Vector3(0.0f, 0.0f, 0.0f);
            }
            else {
                const float len = std::sqrt(len2);
                out[i] = Vector3(x / len, y / len, z / len);
            }
        }
    }

    void transform_ref_novec(const Mat4x4f& m, const Vector3* in, Vector3* out, std::size_t n)
    {
        const float* d = m.data;
        const float m00 = d[0], m10 = d[1], m20 = d[2];
        const float m01 = d[4], m11 = d[5], m21 = d[6];
        const float m02 = d[8], m12 = d[9], m22 = d[10];
        const float m03 = d[12], m13 = d[13], m23 = d[14];
#pragma loop(no_vector)
        for (std::size_t i = 0; i < n; ++i) {
            const float x = in[i].x, y = in[i].y, z = in[i].z;
            out[i] = Vector3(m00 * x + m01 * y + m02 * z + m03,
                m10 * x + m11 * y + m12 * z + m13,
                m20 * x + m21 * y + m22 * z + m23);
        }
    }

} // namespace math::batch