#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace math::scalar {

    // Seuil du "vecteur nul" pour normalized() : |v|^2 <= 1e-12 (|v| < 1e-6) => (0,0,0).
    // DOIT rester identique à kNormalizeEpsSq de la version SIMD (Vector3SIMD.h).
    inline constexpr float kNormalizeEpsSq = 1e-12f;

    // Version C++ pure (aucun intrinsic) : sert de RÉFÉRENCE.
    // Layout : 3 floats contigus, 12 octets, pas de padding (contrairement à math::simd::Vector3).
    // Même sémantique que la version SIMD :
    //  - dot = (x*x' + y*y') + z*z'  (même ordre d'addition)
    //  - normalized : (0,0,0) si |v|^2 <= kNormalizeEpsSq ou NaN, sinon v / sqrt(|v|^2)
    template <typename T = float>
    class Vector3 {
    public:
        T x{}, y{}, z{};

        Vector3() = default;
        Vector3(T x, T y, T z) : x(x), y(y), z(z) {}

        // --- Factory / Constantes ---
        static Vector3 up() { return Vector3(0, 1, 0); }
        static Vector3 down() { return Vector3(0, -1, 0); }
        static Vector3 left() { return Vector3(-1, 0, 0); }
        static Vector3 right() { return Vector3(1, 0, 0); }
        static Vector3 forward() { return Vector3(0, 0, 1); }
        static Vector3 back() { return Vector3(0, 0, -1); }
        static Vector3 one() { return Vector3(1, 1, 1); }
        static Vector3 zero() { return Vector3(0, 0, 0); }
        static Vector3 positiveInfinity() {
            const T inf = std::numeric_limits<T>::infinity();
            return Vector3(inf, inf, inf);
        }
        static Vector3 negativeInfinity() {
            const T inf = std::numeric_limits<T>::infinity();
            return Vector3(-inf, -inf, -inf);
        }

        // --- Opérateurs ---
        Vector3 operator+(const Vector3& o) const { return { x + o.x, y + o.y, z + o.z }; }
        Vector3 operator-(const Vector3& o) const { return { x - o.x, y - o.y, z - o.z }; }
        Vector3 operator*(T s) const { return { x * s, y * s, z * s }; }
        Vector3 operator/(T s) const { return { x / s, y / s, z / s }; }

        Vector3& operator+=(const Vector3& o) { x += o.x; y += o.y; z += o.z; return *this; }
        Vector3& operator-=(const Vector3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
        Vector3& operator*=(T s) { x *= s; y *= s; z *= s; return *this; }
        Vector3& operator/=(T s) { x /= s; y /= s; z /= s; return *this; }

        // --- Produit scalaire, vectoriel, magnitudes ---
        T dot(const Vector3& o) const { return x * o.x + y * o.y + z * o.z; }

        Vector3 cross(const Vector3& o) const {
            return { y * o.z - z * o.y,
                     z * o.x - x * o.z,
                     x * o.y - y * o.x };
        }

        T sqrMagnitude() const { return dot(*this); }
        T magnitude() const { return std::sqrt(sqrMagnitude()); }

        Vector3 normalized() const {
            const T len2 = sqrMagnitude();
            // Écrit sous cette forme pour que NaN => vecteur nul (comme cmpgt côté SIMD)
            if (!(len2 > static_cast<T>(kNormalizeEpsSq))) return zero();
            const T len = std::sqrt(len2);
            return { x / len, y / len, z / len };
        }

        // --- Méthodes statiques géométriques ---
        static T distance(const Vector3& a, const Vector3& b) { return (a - b).magnitude(); }

        static Vector3 Lerp(const Vector3& a, const Vector3& b, T t) {
            t = std::clamp(t, T(0), T(1));
            return LerpUnclamped(a, b, t);
        }

        static Vector3 LerpUnclamped(const Vector3& a, const Vector3& b, T t) {
            return a + (b - a) * t;
        }

        static Vector3 Max(const Vector3& a, const Vector3& b) {
            return { std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z) };
        }

        static Vector3 Min(const Vector3& a, const Vector3& b) {
            return { std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z) };
        }

        static Vector3 MoveTowards(const Vector3& current, const Vector3& target, T maxDelta) {
            Vector3 delta = target - current;
            T sqrDist = delta.sqrMagnitude();
            if (sqrDist <= maxDelta * maxDelta || sqrDist == T(0)) return target;
            T dist = std::sqrt(sqrDist);
            return current + delta * (maxDelta / dist);
        }

        // --- Utilitaires ---
        Vector3 Reflect(const Vector3& normal) const {
            return *this - normal * (T(2) * dot(normal));
        }

        Vector3 Scale(const Vector3& o) const { return { x * o.x, y * o.y, z * o.z }; }

        Vector3 ClampMagnitude(T maxVal) const {
            T sqrMag = sqrMagnitude();
            if (sqrMag > maxVal * maxVal) {
                T s = maxVal / std::sqrt(sqrMag);
                return *this * s;
            }
            return *this;
        }

        void SetVector3(T nx, T ny, T nz) { x = nx; y = ny; z = nz; }

        // --- Accès et comparaisons ---
        T operator[](int index) const {
            if (index == 0) return x;
            if (index == 1) return y;
            if (index == 2) return z;
            throw std::out_of_range("Index out of range");
        }

        bool operator==(const Vector3& o) const { return x == o.x && y == o.y && z == o.z; }
        bool operator!=(const Vector3& o) const { return !(*this == o); }

        std::string toString() const {
            return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
        }

        void print() const { std::cout << "(" << x << ", " << y << ", " << z << ")\n"; }

        std::array<T, 3> toArray() const { return { x, y, z }; }
    };

} // namespace math::scalar