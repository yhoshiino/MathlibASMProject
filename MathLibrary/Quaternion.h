//#pragma once
//#include <cmath>
//#include <string>
//#include <algorithm>
//#include "Vector3.h"
//
//#ifndef M_PI
//#define M_PI 3.14159265358979323846
//#endif
//
//namespace math {
//
//    template<typename T>
//    class Quaternion {
//    public:
//        T w{}, x{}, y{}, z{};
//
//        // Constructeurs
//        Quaternion() : w(1), x(0), y(0), z(0) {}
//        Quaternion(T w, T x, T y, T z) : w(w), x(x), y(y), z(z) {}
//
//        // Identity
//        static Quaternion<T> identity() {
//            return Quaternion<T>(1, 0, 0, 0);
//        }
//
//        // From axis-angle (angle in radians)
//        static Quaternion<T> fromAxisAngle(const Vector3<T>& axis, T angle) {
//            T half = angle * static_cast<T>(0.5);
//            T s = std::sin(half);
//            return Quaternion<T>(std::cos(half), axis.x * s, axis.y * s, axis.z * s);
//        }
//
//        // AngleAxis: angle in degrees
//        static Quaternion<T> AngleAxis(T angleDegrees, const Vector3<T>& axis) {
//            T angleRad = angleDegrees * static_cast<T>(M_PI / 180.0);
//            return fromAxisAngle(axis.normalized(), angleRad);
//        }
//
//        // Dot product between quaternions
//        static T dot(const Quaternion<T>& a, const Quaternion<T>& b) {
//            return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
//        }
//
//        // Magnitude / normalization
//        T magnitude() const {
//            return std::sqrt(w * w + x * x + y * y + z * z);
//        }
//
//        Quaternion<T> normalized() const {
//            T mag = magnitude();
//            return (mag == 0) ? Quaternion<T>() : Quaternion<T>(w / mag, x / mag, y / mag, z / mag);
//        }
//
//        // Conjugate
//        Quaternion<T> conjugate() const {
//            return Quaternion<T>(w, -x, -y, -z);
//        }
//
//        // Inverse
//        static Quaternion<T> inverse(const Quaternion<T>& q) {
//            T magSq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
//            if (magSq == 0) return Quaternion<T>();
//            return Quaternion<T>(q.w / magSq, -q.x / magSq, -q.y / magSq, -q.z / magSq);
//        }
//
//        // Multiply (combine rotations)
//        Quaternion<T> operator*(const Quaternion<T>& q) const {
//            return Quaternion<T>(
//                w * q.w - x * q.x - y * q.y - z * q.z,
//                w * q.x + x * q.w + y * q.z - z * q.y,
//                w * q.y - x * q.z + y * q.w + z * q.x,
//                w * q.z + x * q.y - y * q.x + z * q.w
//            );
//        }
//
//        // Rotate vector (assumes this represents a rotation; uses conjugate)
//        Vector3<T> rotate(const Vector3<T>& v) const {
//            Quaternion<T> qn = this->normalized();
//            Quaternion<T> qv(0, v.x, v.y, v.z);
//            Quaternion<T> res = qn * qv * qn.conjugate();
//            return Vector3<T>(res.x, res.y, res.z);
//        }
//
//        // Angle between two rotations (degrees, 0..180)
//        static T angle(const Quaternion<T>& a, const Quaternion<T>& b) {
//            T d = std::abs(dot(a.normalized(), b.normalized()));
//            d = std::min<T>(d, static_cast<T>(1));
//            // angle = 2 * acos(d) in radians -> degrees
//            return static_cast<T>(2.0) * std::acos(d) * static_cast<T>(180.0 / M_PI);
//        }
//
//        // Lerp (clamped) then normalize
//        static Quaternion<T> lerp(const Quaternion<T>& a, const Quaternion<T>& b, T t) {
//            t = std::max<T>(static_cast<T>(0), std::min<T>(static_cast<T>(1), t));
//            Quaternion<T> res(
//                a.w + (b.w - a.w) * t,
//                a.x + (b.x - a.x) * t,
//                a.y + (b.y - a.y) * t,
//                a.z + (b.z - a.z) * t
//            );
//            return res.normalized();
//        }
//
//        // LerpUnclamped then normalize
//        static Quaternion<T> lerpUnclamped(const Quaternion<T>& a, const Quaternion<T>& b, T t) {
//            Quaternion<T> res(
//                a.w + (b.w - a.w) * t,
//                a.x + (b.x - a.x) * t,
//                a.y + (b.y - a.y) * t,
//                a.z + (b.z - a.z) * t
//            );
//            return res.normalized();
//        }
//
//        // Slerp (spherical linear interpolation)
//        static Quaternion<T> slerp(const Quaternion<T>& a, const Quaternion<T>& b, T t) {
//            T cosTheta = dot(a, b);
//            Quaternion<T> bCopy = b;
//
//            if (cosTheta < 0) {
//                bCopy = Quaternion<T>(-b.w, -b.x, -b.y, -b.z);
//                cosTheta = -cosTheta;
//            }
//
//            if (cosTheta > static_cast<T>(0.9995)) {
//                // Very close -> fallback to lerp
//                return lerp(a, bCopy, t);
//            }
//
//            T theta = std::acos(cosTheta);
//            T sinTheta = std::sin(theta);
//            T w1 = std::sin((1 - t) * theta) / sinTheta;
//            T w2 = std::sin(t * theta) / sinTheta;
//
//            return Quaternion<T>(
//                a.w * w1 + bCopy.w * w2,
//                a.x * w1 + bCopy.x * w2,
//                a.y * w1 + bCopy.y * w2,
//                a.z * w1 + bCopy.z * w2
//            );
//        }
//
//        static Quaternion<T> slerpUnclamped(const Quaternion<T>& a, const Quaternion<T>& b, T t) {
//            // same as slerp but without clamping t; implementation identical
//            return slerp(a, b, t);
//        }
//
//        // Equals exact
//        bool Equals(const Quaternion<T>& q) const {
//            return (w == q.w && x == q.x && y == q.y && z == q.z);
//        }
//
//        bool operator==(const Quaternion<T>& q) const {
//            return Equals(q);
//        }
//
//        // To string
//        std::string toString() const {
//            return "(" + std::to_string(w) + ", " + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
//        }
//
//        // Access components by index: [0]=w [1]=x [2]=y [3]=z
//        T operator[](int index) const {
//            switch (index) {
//            case 0: return w;
//            case 1: return x;
//            case 2: return y;
//            case 3: return z;
//            default: throw std::out_of_range("Quaternion index out of range");
//            }
//        }
//
//        // Set components
//        void Set(T nw, T nx, T ny, T nz) {
//            w = nw; x = nx; y = ny; z = nz;
//        }
//    };
//
//} // namespace math
