#pragma once
#include <cmath>
#include <iostream>
#include <string>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

namespace math {


    class Vector2;

    template<typename T>
    class alignas(16) Vector3 {
    public:
        union {
            __m128 reg;
            struct { float x, y, z; };
        };

        Vector3() : reg(_mm_setzero_ps()) {}
        Vector3(float x, float y, float z) : reg(_mm_setr_ps(x, y, z, 0.0f)) {}
        Vector3(__m128 m) : reg(m) {}

        /*Vector2<T> toVector2() const {
            return Vector2<T>(x, y);
        }*/

        static Vector3 up() { return Vector3(0, 1, 0); }
        static Vector3 down() { return Vector3(0, -1, 0); }
        static Vector3 left() { return Vector3(-1, 0, 0); }
        static Vector3 right() { return Vector3(1, 0, 0); }
        static Vector3 forward() { return Vector3(0, 0, 1); }
        static Vector3 back() { return Vector3(0, 0, -1); }
        static Vector3 one() { return Vector3(1, 1, 1); }
        static Vector3 zero() { return Vector3(0, 0, 0); }
        static Vector3 negativeInfinity() { return Vector3(-INFINITY, -INFINITY, -INFINITY); }
        static Vector3 positiveInfinity() { return Vector3(INFINITY, INFINITY, INFINITY); }

        static T distance(const Vector3& a, const Vector3& b) {
            T dx = a.x - b.x;
            T dy = a.y - b.y;
            T dz = a.z - b.z;
            return sqrt(dx * dx + dy * dy + dz * dz);
        }

        static Vector3<T> Lerp(const Vector3& a, const Vector3& b, float t) {
            if (t < 0) t = 0;
            if (t > 1) t = 1;
            return a + (b - a) * t;
        }

        static Vector3<T> LerpUnclamped(const Vector3& a, const Vector3& b, float t) {
            return a + (b - a) * t;
        }

        static Vector3<T> Max(const Vector3& a, const Vector3& b) {
            return Vector3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
        }

        static Vector3<T> Min(const Vector3& a, const Vector3& b) {
            return Vector3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
        }

        static Vector3<T> MoveTowards(const Vector3& current, const Vector3& target, float maxDelta) {
            Vector3<T> delta = target - current;
            float sqrDist = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
            if (sqrDist <= maxDelta * maxDelta) return target;
            float dist = sqrt(sqrDist);
            return current + delta * (maxDelta / dist);
        }

        Vector3<T> Reflect(const Vector3& normal) const {
            return *this - normal * (2 * this->dot(normal));
        }

        Vector3<T> Scale(const Vector3& other) const {
            return Vector3(x * other.x, y * other.y, z * other.z);
        }

        Vector3<T> ClampMagnitude(const Vector3<T>& v, T max) const {
            T sqrMag = v.x * v.x + v.y * v.y + v.z * v.z;
            if (sqrMag > max * max) {
                T mag = max / sqrt(sqrMag);
                return Vector3<T>{v.x* mag, v.y* mag, v.z* mag};
            }
            return v;
        }

        Vector3 operator+(const Vector3& o) const { return Vector3{ x + o.x, y + o.y, z + o.z }; }
        Vector3 operator-(const Vector3& o) const { return Vector3{ x - o.x, y - o.y, z - o.z }; }
        Vector3 operator*(T scalar) const { return Vector3{ x * scalar, y * scalar, z * scalar }; }
        Vector3 operator/(T divide) const { return Vector3{ x / divide, y / divide, z / divide }; }

        T dot(const Vector3& o) const { return x * o.x + y * o.y + z * o.z; }
        T magnitude() const { return sqrt(x * x + y * y + z * z); }
        T sqrMagnitude() const { return x * x + y * y + z * z; }

        T operator[](int index) const {
            if (index == 0) return x;
            else if (index == 1) return y;
            else if (index == 2) return z;
            else throw std::out_of_range("Index out of range");
        }

        Vector3 normalized() const {
            T mag = magnitude();
            return mag ? Vector3{ x / mag, y / mag, z / mag } : Vector3{ 0,0,0 };
        }

        bool operator==(const Vector3& o) const { return x == o.x && y == o.y && z == o.z; }

        std::string toString() const {
            return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
        }

        void SetVector3(float newX, float newY, float newZ) {
            x = newX;
            y = newY;
            z = newZ;
        }

        void print() const {
            cout << "(" << x << ", " << y << ", " << z << ")\n";
        }

        std::array<T, 3> toArray() const {
            return { x, y, z };
        }
    };
}
