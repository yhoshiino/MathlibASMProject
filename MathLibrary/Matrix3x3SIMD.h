#pragma once
#include <array>
#include <cmath>
#include <iostream>
#include <immintrin.h>
#include "Vector2.h"

namespace math {

    class Mat3x3 {
    public:
        alignas(16) float data[12];

        // Identité
        Mat3x3() {
            _mm_store_ps(data + 0, _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f));
            _mm_store_ps(data + 4, _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f));
            _mm_store_ps(data + 8, _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f));
        }

        float& at(int row, int col) {
            return data[col * 4 + row];
        }

        const float& at(int row, int col) const {
            return data[col * 4 + row];
        }

        static Mat3x3 identity() {
            return Mat3x3();
        }

        // --- Accès aux colonnes (registres SIMD) ---
        __m128 getColumn(int col) const {
            return _mm_load_ps(data + col * 4);
        }


        void setColumn(int col, __m128 v) {
            const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(-1, -1, -1, 0));
            _mm_store_ps(data + col * 4, _mm_and_ps(v, mask));
        }

        // Déterminant = produit mixte des colonnes : c0 . (c1 x c2)
        float determinant() const {
            __m128 c0 = getColumn(0);
            __m128 c1 = getColumn(1);
            __m128 c2 = getColumn(2);

            // Produit vectoriel c1 x c2
            __m128 a_yzx = _mm_shuffle_ps(c1, c1, _MM_SHUFFLE(3, 0, 2, 1));
            __m128 a_zxy = _mm_shuffle_ps(c1, c1, _MM_SHUFFLE(3, 1, 0, 2));
            __m128 b_yzx = _mm_shuffle_ps(c2, c2, _MM_SHUFFLE(3, 0, 2, 1));
            __m128 b_zxy = _mm_shuffle_ps(c2, c2, _MM_SHUFFLE(3, 1, 0, 2));
            __m128 cross = _mm_sub_ps(_mm_mul_ps(a_yzx, b_zxy),
                _mm_mul_ps(a_zxy, b_yzx));

            // Produit scalaire avec c0, puis somme horizontale
            // (la 4e lane vaut 0 grâce au padding)
            __m128 prod = _mm_mul_ps(c0, cross);
            __m128 hi = _mm_movehl_ps(prod, prod);
            __m128 sum = _mm_add_ps(prod, hi);
            __m128 y = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 1, 1, 1));
            sum = _mm_add_ss(sum, y);
            return _mm_cvtss_f32(sum);
        }

        // Matrice x vecteur (registre) : v.x * c0 + v.y * c1 + v.z * c2
        // La 4e lane du vecteur d'entrée est ignorée, celle du résultat vaut 0.
        __m128 multiply(__m128 v) const {
            __m128 vx = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
            __m128 vy = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
            __m128 vz = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));

            __m128 r = _mm_mul_ps(getColumn(0), vx);
            r = _mm_add_ps(r, _mm_mul_ps(getColumn(1), vy));
            r = _mm_add_ps(r, _mm_mul_ps(getColumn(2), vz));
            return r;
        }

        // Matrice x vecteur (tableau de 3 floats)
        std::array<float, 3> multiply(const std::array<float, 3>& vec) const {
            alignas(16) float out[4];
            _mm_store_ps(out, multiply(_mm_setr_ps(vec[0], vec[1], vec[2], 0.0f)));
            return { out[0], out[1], out[2] };
        }

        // Matrice x matrice : chaque colonne du résultat = this * colonne de other
        Mat3x3 operator*(const Mat3x3& other) const {
            Mat3x3 result;
            _mm_store_ps(result.data + 0, multiply(other.getColumn(0)));
            _mm_store_ps(result.data + 4, multiply(other.getColumn(1)));
            _mm_store_ps(result.data + 8, multiply(other.getColumn(2)));
            return result;
        }

        static Mat3x3 rotation(float angle) {
            float c = std::cos(angle);
            float s = std::sin(angle);
            Mat3x3 m;
            _mm_store_ps(m.data + 0, _mm_setr_ps(c, s, 0.0f, 0.0f));
            _mm_store_ps(m.data + 4, _mm_setr_ps(-s, c, 0.0f, 0.0f));
            _mm_store_ps(m.data + 8, _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f));
            return m;
        }

        static Mat3x3 scale(float sx, float sy) {
            Mat3x3 m;
            _mm_store_ps(m.data + 0, _mm_setr_ps(sx, 0.0f, 0.0f, 0.0f));
            _mm_store_ps(m.data + 4, _mm_setr_ps(0.0f, sy, 0.0f, 0.0f));
            _mm_store_ps(m.data + 8, _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f));
            return m;
        }

        static Mat3x3 translate(float tx, float ty) {
            Mat3x3 m;
            _mm_store_ps(m.data + 0, _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f));
            _mm_store_ps(m.data + 4, _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f));
            _mm_store_ps(m.data + 8, _mm_setr_ps(tx, ty, 1.0f, 0.0f));
            return m;
        }

        void print() const {
            for (int row = 0; row < 3; ++row) {
                std::cout << "| ";
                for (int col = 0; col < 3; ++col) {
                    std::cout << at(row, col) << " ";
                }
                std::cout << "|\n";
            }
        }
    };

    // Transforme un point 2D (x, y, 1)
    inline Vector2<float> operator*(const Mat3x3& m, const Vector2<float>& v) {
        alignas(16) float out[4];
        _mm_store_ps(out, m.multiply(_mm_setr_ps(v.x, v.y, 1.0f, 0.0f)));
        return { out[0], out[1] };
    }
}