#pragma once
#include <cmath>
#include <iostream>
#include <string>
#include <algorithm>
#include <limits>
#include <smmintrin.h> // SSE4.1 (pour _mm_dp_ps)

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace math {

    // Forward declaration
    class Vector3;

    class alignas(16) Vector2 {
    public:
        union {
            __m128 reg;
            struct { float x, y; };
        };

        // --- Constructeurs ---
        Vector2() : reg(_mm_setzero_ps()) {}
        Vector2(float x, float y) : reg(_mm_setr_ps(x, y, 0.0f, 0.0f)) {}
        Vector2(__m128 m) : reg(m) {}

        // --- Conversion vers Vector3 ---
        // Définition séparée si Vector3 est déclaré plus bas
        inline Vector3 toVector3(float z = 0.0f) const;

        // --- Constantes / Factory ---
        static Vector2 zero() { return Vector2(_mm_setzero_ps()); }
        static Vector2 up() { return Vector2(0.0f, 1.0f); }
        static Vector2 down() { return Vector2(0.0f, -1.0f); }
        static Vector2 left() { return Vector2(-1.0f, 0.0f); }
        static Vector2 right() { return Vector2(1.0f, 0.0f); }
        static Vector2 one() { return Vector2(1.0f, 1.0f); }
        static Vector2 positiveInfinity() { return Vector2(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()); }
        static Vector2 negativeInfinity() { return Vector2(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()); }

        // --- Opérateurs SIMD de base ---
        Vector2 operator+(const Vector2& o) const { return _mm_add_ps(reg, o.reg); }
        Vector2 operator-(const Vector2& o) const { return _mm_sub_ps(reg, o.reg); }
        Vector2 operator*(float scalar)     const { return _mm_mul_ps(reg, _mm_set1_ps(scalar)); }
        Vector2 operator/(float scalar)     const { return _mm_div_ps(reg, _mm_set1_ps(scalar)); }

        Vector2& operator+=(const Vector2& o) { reg = _mm_add_ps(reg, o.reg); return *this; }
        Vector2& operator-=(const Vector2& o) { reg = _mm_sub_ps(reg, o.reg); return *this; }
        Vector2& operator*=(float scalar) { reg = _mm_mul_ps(reg, _mm_set1_ps(scalar)); return *this; }
        Vector2& operator/=(float scalar) { reg = _mm_div_ps(reg, _mm_set1_ps(scalar)); return *this; }

        // --- Produit scalaire & Magnitudes ---
        // _mm_dp_ps avec le masque 0x33 effectue (x1*x2 + y1*y2) et le place dans x et y
        float dot(const Vector2& o) const {
            return _mm_cvtss_f32(_mm_dp_ps(reg, o.reg, 0x31));
        }

        float sqrMagnitude() const {
            return dot(*this);
        }

        float magnitude() const {
            return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(reg, reg, 0x31)));
        }

        Vector2 normalized() const {
            __m128 dot_reg = _mm_dp_ps(reg, reg, 0x33); // Repète le produit scalaire sur x et y
            __m128 inv_len = _mm_rsqrt_ps(dot_reg);       // Inverse de la racine carrée (rapide)

            // Si la norme est 0, évite la division par zéro / NaN
            if (_mm_cvtss_f32(dot_reg) <= 0.00001f) return zero();
            return _mm_mul_ps(reg, inv_len);
        }

        // --- Opérations Géométriques Static ---
        static float distance(const Vector2& a, const Vector2& b) {
            return (a - b).magnitude();
        }

        static Vector2 Min(const Vector2& a, const Vector2& b) {
            return _mm_min_ps(a.reg, b.reg);
        }

        static Vector2 Max(const Vector2& a, const Vector2& b) {
            return _mm_max_ps(a.reg, b.reg);
        }

        static Vector2 Lerp(const Vector2& a, const Vector2& b, float t) {
            t = std::clamp(t, 0.0f, 1.0f);
            return LerpUnclamped(a, b, t);
        }

        static Vector2 LerpUnclamped(const Vector2& a, const Vector2& b, float t) {
            // a + (b - a) * t
            __m128 vt = _mm_set1_ps(t);
            return _mm_add_ps(a.reg, _mm_mul_ps(_mm_sub_ps(b.reg, a.reg), vt));
        }

        static float angle(const Vector2& a, const Vector2& b) {
            float magP = a.magnitude() * b.magnitude();
            if (magP == 0.0f) return 0.0f;

            float cosTheta = std::clamp(a.dot(b) / magP, -1.0f, 1.0f);
            return std::acos(cosTheta) * (180.0f / M_PI);
        }

        static float SignedAngle(const Vector2& from, const Vector2& to) {
            float angle = std::atan2(to.y, to.x) - std::atan2(from.y, from.x);
            return angle * (180.0f / M_PI);
        }

        static Vector2 MoveTowards(const Vector2& current, const Vector2& target, float maxDelta) {
            Vector2 delta = target - current;
            float sqrDist = delta.sqrMagnitude();
            if (sqrDist <= maxDelta * maxDelta || sqrDist == 0.0f) return target;

            float dist = std::sqrt(sqrDist);
            return current + delta * (maxDelta / dist);
        }

        static Vector2 SmoothDamp(const Vector2& current, Vector2 target, Vector2& velocity, float smoothTime, float deltaTime) {
            smoothTime = std::max(0.0001f, smoothTime);
            float omega = 2.0f / smoothTime;
            float x = omega * deltaTime;
            float exp = 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);

            Vector2 change = current - target;
            Vector2 temp = (velocity + change * omega) * deltaTime;
            velocity = (velocity - temp * omega) * exp;

            return target + (change + temp) * exp;
        }

        // --- Utilitaires ---
        Vector2 Perpendicular() const {
            return Vector2(-y, x);
        }

        Vector2 Reflect(const Vector2& normal) const {
            return *this - normal * (2.0f * this->dot(normal));
        }

        Vector2 Scale(const Vector2& other) const {
            return _mm_mul_ps(reg, other.reg);
        }

        Vector2 ClampMagnitude(float max) const {
            float sqrMag = sqrMagnitude();
            if (sqrMag > max * max) {
                float scale = max / std::sqrt(sqrMag);
                return *this * scale;
            }
            return *this;
        }

        // --- Accès et Comparaisons ---
        float operator[](int index) const {
            if (index == 0) return x;
            if (index == 1) return y;
            throw std::out_of_range("Index out of range");
        }

        bool operator==(const Vector2& o) const {
            // Compare les registres SSE (masque toutes les composantes)
            __m128 cmp = _mm_cmpeq_ps(reg, o.reg);
            return (_mm_movemask_ps(cmp) & 0x3) == 0x3; // Vérifie X et Y
        }

        std::string toString() const {
            return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
        }

        void print() const {
            std::cout << "(" << x << ", " << y << ")\n";
        }
    };
}