#pragma once
#include <array>
#include <cmath>
#include <iostream>
#include <immintrin.h> // SSE / AVX / FMA
#include "Vector2.h"

namespace math {

    // ------------------------------------------------------------------
    // Version générique (double, int, ...) : ton code d'origine, inchangé
    // ------------------------------------------------------------------
    template<typename T = float>
    class Mat3x3 {
    public:
        // Column-major storage: 3 columns of 3 elements each
        std::array<T, 9> data;

        Mat3x3() {
            data = {
                1, 0, 0,  // column 0
                0, 1, 0,  // column 1
                0, 0, 1   // column 2
            };
        }

        T& at(int row, int col) { return data[col * 3 + row]; }
        const T& at(int row, int col) const { return data[col * 3 + row]; }

        static Mat3x3<T> identity() { return Mat3x3<T>(); }

        std::array<T, 3> multiply(const std::array<T, 3>& vec) const {
            std::array<T, 3> result = { 0, 0, 0 };
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col)
                    result[row] += at(row, col) * vec[col];
            return result;
        }

        Mat3x3 operator*(const Mat3x3& other) const {
            Mat3x3 result;
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    result.at(row, col) = 0;
                    for (int k = 0; k < 3; ++k)
                        result.at(row, col) += at(row, k) * other.at(k, col);
                }
            }
            return result;
        }

        static Mat3x3 rotation(T angle) {
            T c = std::cos(angle);
            T s = std::sin(angle);
            Mat3x3 rot;
            rot.at(0, 0) = c;  rot.at(0, 1) = -s;
            rot.at(1, 0) = s;  rot.at(1, 1) = c;
            rot.at(2, 2) = 1;
            return rot;
        }

        static Mat3x3 scale(T sx, T sy) {
            Mat3x3 m;
            m.at(0, 0) = sx;
            m.at(1, 1) = sy;
            m.at(2, 2) = 1;
            return m;
        }

        static Mat3x3 translate(T tx, T ty) {
            Mat3x3 m;
            m.at(0, 2) = tx;
            m.at(1, 2) = ty;
            return m;
        }

        void print() const {
            for (int row = 0; row < 3; ++row) {
                std::cout << "| ";
                for (int col = 0; col < 3; ++col)
                    std::cout << at(row, col) << " ";
                std::cout << "|\n";
            }
        }
    };

    template<typename T>
    Vector2<T> operator*(const Mat3x3<T>& m, const Vector2<T>& v) {
        return {
            m.at(0, 0) * v.x + m.at(0, 1) * v.y + m.at(0, 2),
            m.at(1, 0) * v.x + m.at(1, 1) * v.y + m.at(1, 2)
        };
    }

    // ------------------------------------------------------------------
    // Version SIMD (SSE) : spécialisation pour float
    //
    // Stockage : 3 colonnes de 4 floats (la 4e lane est du padding, toujours 0),
    // soit 12 floats alignés sur 16 octets. Chaque colonne tient dans un __m128.
    // at(row, col) = data[col * 4 + row]
    // ------------------------------------------------------------------
    namespace detail {
        // a * b + c (une seule instruction si FMA est disponible)
        inline __m128 madd(__m128 a, __m128 b, __m128 c) {
#if defined(__FMA__) || defined(__AVX2__)
            return _mm_fmadd_ps(a, b, c);
#else
            return _mm_add_ps(_mm_mul_ps(a, b), c);
#endif
        }
    }

    template<>
    class alignas(16) Mat3x3<float> {
    public:
        alignas(16) float data[12]; // 3 colonnes x 4 floats (padding en lane 3)

        Mat3x3() {
            _mm_store_ps(data + 0, _mm_setr_ps(1, 0, 0, 0)); // colonne 0
            _mm_store_ps(data + 4, _mm_setr_ps(0, 1, 0, 0)); // colonne 1
            _mm_store_ps(data + 8, _mm_setr_ps(0, 0, 1, 0)); // colonne 2
        }

        float& at(int row, int col) { return data[col * 4 + row]; }
        const float& at(int row, int col) const { return data[col * 4 + row]; }

        // Accès direct aux colonnes en registre SIMD
        __m128 getColumn(int col) const { return _mm_load_ps(data + col * 4); }
        void setColumn(int col, __m128 v) { _mm_store_ps(data + col * 4, v); }

        static Mat3x3<float> identity() { return Mat3x3<float>(); }

        // M * v pour un __m128 (x, y, z, ignoré). Le résultat a sa lane 3 à 0.
        __m128 multiply(__m128 v) const {
            __m128 x = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
            __m128 y = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
            __m128 z = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));

            __m128 r = _mm_mul_ps(getColumn(0), x);
            r = detail::madd(getColumn(1), y, r);
            r = detail::madd(getColumn(2), z, r);
            return r;
        }

        // Même interface que la version générique
        std::array<float, 3> multiply(const std::array<float, 3>& vec) const {
            __m128 r = multiply(_mm_setr_ps(vec[0], vec[1], vec[2], 0.0f));
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, r);
            return { tmp[0], tmp[1], tmp[2] };
        }

        // Chaque colonne du résultat = M * (colonne correspondante de other)
        Mat3x3 operator*(const Mat3x3& other) const {
            Mat3x3 result;
            for (int col = 0; col < 3; ++col)
                result.setColumn(col, multiply(other.getColumn(col)));
            return result;
        }

        static Mat3x3 rotation(float angle) {
            float c = std::cos(angle);
            float s = std::sin(angle);
            Mat3x3 rot;
            rot.setColumn(0, _mm_setr_ps(c, s, 0, 0));
            rot.setColumn(1, _mm_setr_ps(-s, c, 0, 0));
            rot.setColumn(2, _mm_setr_ps(0, 0, 1, 0));
            return rot;
        }

        static Mat3x3 scale(float sx, float sy) {
            Mat3x3 m;
            m.setColumn(0, _mm_setr_ps(sx, 0, 0, 0));
            m.setColumn(1, _mm_setr_ps(0, sy, 0, 0));
            return m;
        }

        static Mat3x3 translate(float tx, float ty) {
            Mat3x3 m;
            m.setColumn(2, _mm_setr_ps(tx, ty, 1, 0));
            return m;
        }

        void print() const {
            for (int row = 0; row < 3; ++row) {
                std::cout << "| ";
                for (int col = 0; col < 3; ++col)
                    std::cout << at(row, col) << " ";
                std::cout << "|\n";
            }
        }
    };

    // Transformation d'un point 2D (x, y, 1). Non-template : choisi de préférence
    // au template générique ci-dessus quand T = float.
    inline Vector2<float> operator*(const Mat3x3<float>& m, const Vector2<float>& v) {
        __m128 r = m.multiply(_mm_setr_ps(v.x, v.y, 1.0f, 0.0f));
        return {
            _mm_cvtss_f32(r),
            _mm_cvtss_f32(_mm_shuffle_ps(r, r, _MM_SHUFFLE(1, 1, 1, 1)))
        };
    }

} // namespace math