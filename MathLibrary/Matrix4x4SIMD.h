#pragma once
#include <emmintrin.h> // SSE2 uniquement
#include <cmath>

namespace math::simd {

    // Convention : stockage column-major (col[j] = colonne j, data[col*4 + row]),
    // vecteurs colonnes (M * v), translation dans col[3], angles en radians, repère main droite.
    // Composition : A * B applique B en premier, puis A.
    // NB : union avec tableau de __m128 = pratique courante sous MSVC/GCC/Clang.
    struct alignas(16) Mat4x4f {
        union {
            __m128 col[4];     // Accès vectoriel SIMD par colonne
            float data[16];    // Accès plat column-major
        };

        // Identité
        Mat4x4f() {
            col[0] = _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f);
            col[1] = _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f);
            col[2] = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
            col[3] = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
        }

        float& at(int row, int colIdx) { return data[colIdx * 4 + row]; }
        const float& at(int row, int colIdx) const { return data[colIdx * 4 + row]; }

        static Mat4x4f translate(float tx, float ty, float tz) {
            Mat4x4f m;
            m.col[3] = _mm_setr_ps(tx, ty, tz, 1.0f);
            return m;
        }

        static Mat4x4f rotationZ(float angleRad) {
            Mat4x4f m;
            float c = std::cos(angleRad);
            float s = std::sin(angleRad);
            m.col[0] = _mm_setr_ps(c, s, 0.0f, 0.0f);
            m.col[1] = _mm_setr_ps(-s, c, 0.0f, 0.0f);
            return m;
        }

        // Produit M * v complet (4 colonnes pondérées par x, y, z, w de v).
        // Sert de brique à operator*.
        __m128 multiply(__m128 v) const {
            __m128 x = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
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
                r.col[j] = multiply(o.col[j]);
            return r;
        }
    };

} // namespace math::simd