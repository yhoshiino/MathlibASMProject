#pragma once
#include <cmath>
#include <iostream>
#include <string>
#include <algorithm>
#include <limits>
#include <array>
#include <stdexcept>
#include <smmintrin.h> // SSE4.1 (_mm_dp_ps)

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace math {

    // Helper clamp pour compatibilité C++14 / C++11
    template <typename T>
    inline T clamp(T val, T minVal, T maxVal) {
        return std::max(minVal, std::min(val, maxVal));
    }

    // Forward declaration
    class Vector2;

    class alignas(16) Vector3 {
    public:
        union {
            __m128 reg;
            struct { float x, y, z, w; }; // w assure l'alignement de 16 octets du registre
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
        static Vector3 negativeInfinity() { return Vector3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()); }
        static Vector3 positiveInfinity() { return Vector3(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()); }

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
        float dot(const Vector3& o) const {
            return _mm_cvtss_f32(_mm_dp_ps(reg, o.reg, 0x71));
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
            return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(reg, reg, 0x71)));
        }

        Vector3 normalized() const {
            __m128 dot_reg = _mm_dp_ps(reg, reg, 0x77);
            if (_mm_cvtss_f32(dot_reg) <= 0.00001f) return zero();
            return _mm_mul_ps(reg, _mm_rsqrt_ps(dot_reg));
        }

        // --- Méthodes Statiques Géométriques ---
        static float distance(const Vector3& a, const Vector3& b) {
            __m128 diff = _mm_sub_ps(a.reg, b.reg);
            __m128 dot = _mm_dp_ps(diff, diff, 0x71);
            return _mm_cvtss_f32(_mm_sqrt_ss(dot));
        }

        static Vector3 Lerp(const Vector3& a, const Vector3& b, float t) {
            t = math::clamp(t, 0.0f, 1.0f);
            return LerpUnclamped(a, b, t);
        }

        static Vector3 LerpUnclamped(const Vector3& a, const Vector3& b, float t) {
            __m128 vt = _mm_set1_ps(t);
            return _mm_add_ps(a.reg, _mm_mul_ps(_mm_sub_ps(b.reg, a.reg), vt));
        }

        static Vector3 Max(const Vector3& a, const Vector3& b) {
            return _mm_max_ps(a.reg, b.reg);
        }

        static Vector3 Min(const Vector3& a, const Vector3& b) {
            return _mm_min_ps(a.reg, b.reg);
        }

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
            return (_mm_movemask_ps(cmp) & 0x7) == 0x7; // Vérifie X, Y, Z
        }

        bool operator!=(const Vector3& o) const {
            return !(*this == o);
        }

        std::string toString() const {
            return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
        }

        void print() const {
            std::cout << "(" << x << ", " << y << ", " << z << ")\n";
        }

        std::array<float, 3> toArray() const {
            return { x, y, z };
        }
    };
}