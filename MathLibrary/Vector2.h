#pragma once
#include <cmath>
#include <iostream>
#include <string>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

namespace math {
    template<typename T> class Vector3;

    template<typename T>
    class Vector2 {
    public:
        T x{}, y{};

        Vector2() = default;
        Vector2(T x, T y) : x(x), y(y) {}

        Vector3<T> toVector3(T z = 0) {
            return Vector3<T>(x, y, z);
        }

        static Vector2 up() {
            return Vector2(0, 1);
        }
        static Vector2 down() {
            return Vector2(0, -1);
        }
        static Vector2 left() {
            return Vector2(-1, 0);
        }
        static Vector2 right() {
            return Vector2(1, 0);
        }
        static Vector2 one() {
            return Vector2(1, 1);
        }
        static Vector2 negativeInfinity() {
            return Vector2(-INFINITY, -INFINITY);
        }
        static Vector2 positiveInfinity() {
            return Vector2(INFINITY, INFINITY);
        }
        static T distance(const Vector2& a, const Vector2& b) {
            T dx = a.x - b.x;
            T dy = a.y - b.y;
            return sqrt(dx * dx + dy * dy);
        }

        static float angle(const Vector2& a, const Vector2& b) {
            float dotP = a.dot(b);
            float magP = a.magnitude() * b.magnitude();
            if (magP == 0) return 0.0f;


            float cosTheta = dotP / magP;

            if (cosTheta > 1.0f) cosTheta = 1.0f;
            if (cosTheta < -1.0f) cosTheta = -1.0f;

            float angleRad = acos(cosTheta);
            return angleRad * (180.0f / static_cast<float>(M_PI));
        }

        static Vector2<T> Lerp(const Vector2& a, const Vector2& b, float t) {
            if (t < 0) t = 0;
            if (t > 1) t = 1;
            return a + (b - a) * t;
        }

        static Vector2<T> LerpUnclamped(const Vector2& a, const Vector2& b, float t) {
            return a + (b - a) * t;
        }

        static Vector2<T> Max(const Vector2& a, const Vector2& b) {
            return Vector2(std::max(a.x, b.x), std::max(a.y, b.y));
        }

        static Vector2<T> Min(const Vector2& a, const Vector2& b) {
            return Vector2(std::min(a.x, b.x), std::min(a.y, b.y));
        }

        static Vector2<T> MoveTowards(const Vector2& current, const Vector2& target, float maxDelta) {
            Vector2<T> delta = target - current;
            float sqrDist = delta.x * delta.x + delta.y * delta.y;
            if (sqrDist <= maxDelta * maxDelta) return target;
            float dist = std::sqrt(sqrDist);
            return current + delta * (maxDelta / dist);
        }

        static float SignedAngle(const Vector2& from, const Vector2& to) {
            float angle = std::atan2(to.y, to.x) - std::atan2(from.y, from.x);
            return angle * (180.0f / static_cast<float>(M_PI));
        }

        static Vector2<T> SmoothDamp(const Vector2& current, const Vector2& target, Vector2<T>& velocity, float smoothTime, float deltaTime) {
            float omega = 2.0f / smoothTime;
            float x = omega * deltaTime;
            float exp = 1.0f / (1.0f + x + 0.48f * x * x + 0.235f * x * x * x);

            Vector2<T> change = current - target;
            Vector2<T> temp = (velocity + change * omega) * deltaTime;
            velocity = (velocity - temp * omega) * exp;

            Vector2<T> result = target + (change + temp) * exp;
            return result;
        }

        Vector2<T> Perpendicular() const {
            return Vector2(-y, x);
        }

        Vector2<T> Reflect(const Vector2& normal) const {
            return *this - normal * (2 * this->dot(normal));
        }

        Vector2<T> Scale(const Vector2& other) const {
            return Vector2(x * other.x, y * other.y);
        }

        Vector2<T> Clampmagnitude(const Vector2<T>& v, T max) const {
            T sqrMag = v.x * v.x + v.y * v.y;
            if (sqrMag > max * max) {
                T mag = max / sqrt(sqrMag);
                return Vector2<T>{ v.x* mag, v.y* mag };
            }
            return v;
        }

        Vector2 operator+(const Vector2& o) const {
            return Vector2{ x + o.x, y + o.y };
        }
        Vector2 operator-(const Vector2& o) const {
            return Vector2{ x - o.x, y - o.y };
        }
        Vector2 operator*(T scalar) const {
            return Vector2{ x * scalar, y * scalar };
        }
        Vector2 operator/(T divide) const {
            return Vector2{ x / divide, y / divide };
        }

        Vector2<T>& operator+=(const Vector2<T>& other) {
            x += other.x;
            y += other.y;
            return *this;
        }

        Vector2<T>& operator-=(const Vector2<T>& other) {
            x -= other.x;
            y -= other.y;
            return *this;
        }

        Vector2<T>& operator*=(T scalar) {
            x *= scalar;
            y *= scalar;
            return *this;
        }

        Vector2<T>& operator/=(T scalar) {
            x /= scalar;
            y /= scalar;
            return *this;
        }

        T dot(const Vector2& o) const {
            return x * o.x + y * o.y;
        }

        T magnitude() const {
            return sqrt(x * x + y * y);
        }

        T sqrMagnitude() const {
            return x * x + y * y;
        }

        T operator[](int index) const {
            if (index == 0) return x;
            else if (index == 1) return y;
            else throw std::out_of_range("Index out of range");
        }

        Vector2 normalized() const {
            T mag = magnitude();
            return mag ? Vector2{ x / mag, y / mag } : Vector2{ 0,0 };
        }

        bool operator==(const Vector2& o) const {
            return x == o.x && y == o.y;
        }

        std::string toString() const {
            return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
        }




        void SetVector2(float newX, float newY) {
            x = newX;
            y = newY;
        }

        void print() const {
            cout << "(" << x << ", " << y << ")\n";
        }
    };

}