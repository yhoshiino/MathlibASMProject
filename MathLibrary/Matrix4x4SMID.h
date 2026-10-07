#pragma once
#include <immintrin.h>

namespace math {

    struct alignas(16) Mat4x4f {
        __m128 col[4]; // col[j] = colonne j

        Mat4x4f() {
            col[0] = _mm_setr_ps(1, 0, 0, 0);
            col[1] = _mm_setr_ps(0, 1, 0, 0);
            col[2] = _mm_setr_ps(0, 0, 1, 0);
            col[3] = _mm_setr_ps(0, 0, 0, 1);
        }

        static Mat4x4f identity() { return Mat4x4f(); }

        static Mat4x4f translate(float tx, float ty, float tz) {
            Mat4x4f m;
            m.col[3] = _mm_setr_ps(tx, ty, tz, 1.0f);
            return m;
        }

        static Mat4x4f scale(float sx, float sy, float sz) {
            Mat4x4f m;
            m.col[0] = _mm_setr_ps(sx, 0, 0, 0);
            m.col[1] = _mm_setr_ps(0, sy, 0, 0);
            m.col[2] = _mm_setr_ps(0, 0, sz, 0);
            return m;
        }

        // r = M * v
        __m128 multiply(__m128 v) const {
            __m128 x = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0)); // v.x partout
            __m128 y = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
            __m128 z = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
            __m128 w = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));

            __m128 r = _mm_mul_ps(col[0], x);
            r = _mm_add_ps(r, _mm_mul_ps(col[1], y));
            r = _mm_add_ps(r, _mm_mul_ps(col[2], z));
            r = _mm_add_ps(r, _mm_mul_ps(col[3], w));
            return r;
        }

        Mat4x4f operator*(const Mat4x4f& o) const {
            Mat4x4f r;
            for (int j = 0; j < 4; ++j)
                r.col[j] = multiply(o.col[j]); // M * (colonne j de o)
            return r;
        }

        Mat4x4f transpose() const {
            __m128 t0 = _mm_unpacklo_ps(col[0], col[1]); // c0.x c1.x c0.y c1.y
            __m128 t1 = _mm_unpacklo_ps(col[2], col[3]); // c2.x c3.x c2.y c3.y
            __m128 t2 = _mm_unpackhi_ps(col[0], col[1]); // c0.z c1.z c0.w c1.w
            __m128 t3 = _mm_unpackhi_ps(col[2], col[3]); // c2.z c3.z c2.w c3.w

            Mat4x4f r;
            r.col[0] = _mm_movelh_ps(t0, t1); // c0.x c1.x c2.x c3.x
            r.col[1] = _mm_movehl_ps(t1, t0); // c0.y c1.y c2.y c3.y
            r.col[2] = _mm_movelh_ps(t2, t3);
            r.col[3] = _mm_movehl_ps(t3, t2);
            return r;
        }
    };

} // namespace math