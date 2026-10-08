#pragma once
#include <algorithm>
#include <array>
#include <cmath>

namespace math::scalar {

    // Convention : stockage column-major data[col*4 + row], vecteurs colonnes (M * v),
    // translation dans la colonne 3, angles en radians, repère main droite.
    // trs = T * Rz * Ry * Rx * S  (le scale est appliqué en premier).
    template<typename T = float>
    class Mat4x4 {
    public:
        std::array<T, 16> data;

        Mat4x4() {
            data = {
                1, 0, 0, 0, //column 0
                0, 1, 0, 0, //column 1
                0, 0, 1, 0, //column 2
                0, 0, 0, 1  //column 3
            };
        }

        T& at(int row, int col) { return data[col * 4 + row]; }
        const T& at(int row, int col) const { return data[col * 4 + row]; }

        static Mat4x4<T> identity() { return Mat4x4<T>(); }

        static Mat4x4<T> zero() {
            Mat4x4<T> m;
            m.data.fill(0);
            return m;
        }

        static Mat4x4<T> translate(T tx, T ty, T tz) {
            Mat4x4<T> m = identity();
            m.at(0, 3) = tx;
            m.at(1, 3) = ty;
            m.at(2, 3) = tz;
            return m;
        }

        static Mat4x4<T> scale(T sx, T sy, T sz) {
            Mat4x4<T> m = identity();
            m.at(0, 0) = sx;
            m.at(1, 1) = sy;
            m.at(2, 2) = sz;
            return m;
        }

        static Mat4x4<T> rotationX(T angle) {
            T c = std::cos(angle);
            T s = std::sin(angle);
            Mat4x4<T> m = identity();
            m.at(1, 1) = c;
            m.at(1, 2) = -s;
            m.at(2, 1) = s;
            m.at(2, 2) = c;
            return m;
        }

        static Mat4x4<T> rotationY(T angle) {
            T c = std::cos(angle);
            T s = std::sin(angle);
            Mat4x4<T> m = identity();
            m.at(0, 0) = c;
            m.at(0, 2) = s;
            m.at(2, 0) = -s;
            m.at(2, 2) = c;
            return m;
        }

        static Mat4x4<T> rotationZ(T angle) {
            T c = std::cos(angle);
            T s = std::sin(angle);
            Mat4x4<T> m = identity();
            m.at(0, 0) = c;
            m.at(0, 1) = -s;
            m.at(1, 0) = s;
            m.at(1, 1) = c;
            return m;
        }

        static Mat4x4<T> trs(T tx, T ty, T tz, T angleX, T angleY, T angleZ, T sx, T sy, T sz) {
            return translate(tx, ty, tz) * rotationZ(angleZ) * rotationY(angleY) * rotationX(angleX) * scale(sx, sy, sz);
        }

        Mat4x4<T> operator*(const Mat4x4<T>& other) const {
            Mat4x4<T> result = zero();
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    for (int k = 0; k < 4; ++k) {
                        result.at(row, col) += at(row, k) * other.at(k, col);
                    }
                }
            }
            return result;
        }

        std::array<T, 4> multiplyPoint(const std::array<T, 4>& vec) const {
            std::array<T, 4> result = { 0, 0, 0, 0 };
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    result[row] += at(row, col) * vec[col];
                }
            }
            return result;
        }

        // Point avec w = 1, sans division perspective (suppose une matrice affine).
        std::array<T, 3> multiplyPoint3x4(const std::array<T, 3>& vec) const {
            std::array<T, 3> result = { 0, 0, 0 };
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    result[row] += at(row, col) * vec[col];
                }
                result[row] += at(row, 3);
            }
            return result;
        }

        // Direction (w = 0) : la translation est ignorée.
        std::array<T, 3> multiplyVector(const std::array<T, 3>& vec) const {
            std::array<T, 3> result = { 0, 0, 0 };
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    result[row] += at(row, col) * vec[col];
                }
            }
            return result;
        }

        std::array<T, 4> getColumn(int col) const {
            return { at(0, col), at(1, col), at(2, col), at(3, col) };
        }

        std::array<T, 4> getRow(int row) const {
            return { at(row, 0), at(row, 1), at(row, 2), at(row, 3) };
        }

        void setColumn(int col, const std::array<T, 4>& values) {
            for (int row = 0; row < 4; ++row) {
                at(row, col) = values[row];
            }
        }

        void setRow(int row, const std::array<T, 4>& values) {
            for (int col = 0; col < 4; ++col) {
                at(row, col) = values[col];
            }
        }

        bool isIdentity() const {
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    T expected = (row == col) ? 1 : 0;
                    if (at(row, col) != expected) return false;
                }
            }
            return true;
        }

        Mat4x4<T> transpose() const {
            Mat4x4<T> result;
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    result.at(row, col) = at(col, row);
                }
            }
            return result;
        }

        T determinant() const {
            T det = 0;
            for (int i = 0; i < 4; ++i) {
                std::array<T, 9> minor;
                int idx = 0;
                for (int row = 1; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        if (col == i) continue;
                        minor[idx++] = at(row, col);
                    }
                }
                T cofactor = minor[0] * (minor[4] * minor[8] - minor[5] * minor[7])
                    - minor[1] * (minor[3] * minor[8] - minor[5] * minor[6])
                    + minor[2] * (minor[3] * minor[7] - minor[4] * minor[6]);
                det += ((i % 2 == 0) ? 1 : -1) * at(0, i) * cofactor;
            }
            return det;
        }

        Mat4x4<T> inverse() const {
            Mat4x4<T> inv;
            T m[16];
            std::copy(data.begin(), data.end(), m);

            T invOut[16];
            invOut[0] = m[5] * m[10] * m[15] -
                m[5] * m[11] * m[14] -
                m[9] * m[6] * m[15] +
                m[9] * m[7] * m[14] +
                m[13] * m[6] * m[11] -
                m[13] * m[7] * m[10];

            invOut[4] = -m[4] * m[10] * m[15] +
                m[4] * m[11] * m[14] +
                m[8] * m[6] * m[15] -
                m[8] * m[7] * m[14] -
                m[12] * m[6] * m[11] +
                m[12] * m[7] * m[10];

            invOut[8] = m[4] * m[9] * m[15] -
                m[4] * m[11] * m[13] -
                m[8] * m[5] * m[15] +
                m[8] * m[7] * m[13] +
                m[12] * m[5] * m[11] -
                m[12] * m[7] * m[9];

            invOut[12] = -m[4] * m[9] * m[14] +
                m[4] * m[10] * m[13] +
                m[8] * m[5] * m[14] -
                m[8] * m[6] * m[13] -
                m[12] * m[5] * m[10] +
                m[12] * m[6] * m[9];

            invOut[1] = -m[1] * m[10] * m[15] +
                m[1] * m[11] * m[14] +
                m[9] * m[2] * m[15] -
                m[9] * m[3] * m[14] -
                m[13] * m[2] * m[11] +
                m[13] * m[3] * m[10];

            invOut[5] = m[0] * m[10] * m[15] -
                m[0] * m[11] * m[14] -
                m[8] * m[2] * m[15] +
                m[8] * m[3] * m[14] +
                m[12] * m[2] * m[11] -
                m[12] * m[3] * m[10];

            invOut[9] = -m[0] * m[9] * m[15] +
                m[0] * m[11] * m[13] +
                m[8] * m[1] * m[15] -
                m[8] * m[3] * m[13] -
                m[12] * m[1] * m[11] +
                m[12] * m[3] * m[9];

            invOut[13] = m[0] * m[9] * m[14] -
                m[0] * m[10] * m[13] -
                m[8] * m[1] * m[14] +
                m[8] * m[2] * m[13] +
                m[12] * m[1] * m[10] -
                m[12] * m[2] * m[9];

            invOut[2] = m[1] * m[6] * m[15] -
                m[1] * m[7] * m[14] -
                m[5] * m[2] * m[15] +
                m[5] * m[3] * m[14] +
                m[13] * m[2] * m[7] -
                m[13] * m[3] * m[6];

            invOut[6] = -m[0] * m[6] * m[15] +
                m[0] * m[7] * m[14] +
                m[4] * m[2] * m[15] -
                m[4] * m[3] * m[14] -
                m[12] * m[2] * m[7] +
                m[12] * m[3] * m[6];

            invOut[10] = m[0] * m[5] * m[15] -
                m[0] * m[7] * m[13] -
                m[4] * m[1] * m[15] +
                m[4] * m[3] * m[13] +
                m[12] * m[1] * m[7] -
                m[12] * m[3] * m[5];

            invOut[14] = -m[0] * m[5] * m[14] +
                m[0] * m[6] * m[13] +
                m[4] * m[1] * m[14] -
                m[4] * m[2] * m[13] -
                m[12] * m[1] * m[6] +
                m[12] * m[2] * m[5];

            invOut[3] = -m[1] * m[6] * m[11] +
                m[1] * m[7] * m[10] +
                m[5] * m[2] * m[11] -
                m[5] * m[3] * m[10] -
                m[9] * m[2] * m[7] +
                m[9] * m[3] * m[6];

            invOut[7] = m[0] * m[6] * m[11] -
                m[0] * m[7] * m[10] -
                m[4] * m[2] * m[11] +
                m[4] * m[3] * m[10] +
                m[8] * m[2] * m[7] -
                m[8] * m[3] * m[6];

            invOut[11] = -m[0] * m[5] * m[11] +
                m[0] * m[7] * m[9] +
                m[4] * m[1] * m[11] -
                m[4] * m[3] * m[9] -
                m[8] * m[1] * m[7] +
                m[8] * m[3] * m[5];

            invOut[15] = m[0] * m[5] * m[10] -
                m[0] * m[6] * m[9] -
                m[4] * m[1] * m[10] +
                m[4] * m[2] * m[9] +
                m[8] * m[1] * m[6] -
                m[8] * m[2] * m[5];

            T det = m[0] * invOut[0] + m[1] * invOut[4] + m[2] * invOut[8] + m[3] * invOut[12];
            if (det == 0) return zero();

            T invDet = T(1) / det;
            for (int i = 0; i < 16; ++i) {
                inv.data[i] = invOut[i] * invDet;
            }
            return inv;
        }

        std::array<T, 3> lossyScale() const {
            std::array<T, 3> s;
            for (int i = 0; i < 3; ++i) {
                s[i] = std::sqrt(
                    at(0, i) * at(0, i) +
                    at(1, i) * at(1, i) +
                    at(2, i) * at(2, i)
                );
            }
            return s;
        }

        std::array<T, 3> getPosition() const {
            return { at(0, 3), at(1, 3), at(2, 3) };
        }

        void setTRS(T tx, T ty, T tz, T angleX, T angleY, T angleZ, T sx, T sy, T sz) {
            *this = trs(tx, ty, tz, angleX, angleY, angleZ, sx, sy, sz);
        }

        bool validTRS() const {
            return std::abs(determinant()) > static_cast<T>(1e-6);
        }

        static Mat4x4<T> perspective(T fovY, T aspect, T zNear, T zFar) {
            T f = 1 / std::tan(fovY / 2);
            Mat4x4<T> m = zero();
            m.at(0, 0) = f / aspect;
            m.at(1, 1) = f;
            m.at(2, 2) = (zFar + zNear) / (zNear - zFar);
            m.at(2, 3) = (2 * zFar * zNear) / (zNear - zFar);
            m.at(3, 2) = -1;
            return m;
        }

        static Mat4x4<T> ortho(T left, T right, T bottom, T top, T zNear, T zFar) {
            Mat4x4<T> m = identity();
            m.at(0, 0) = 2 / (right - left);
            m.at(1, 1) = 2 / (top - bottom);
            m.at(2, 2) = -2 / (zFar - zNear);
            m.at(0, 3) = -(right + left) / (right - left);
            m.at(1, 3) = -(top + bottom) / (top - bottom);
            m.at(2, 3) = -(zFar + zNear) / (zFar - zNear);
            return m;
        }

        static Mat4x4<T> frustum(T left, T right, T bottom, T top, T zNear, T zFar) {
            Mat4x4<T> m = zero();
            m.at(0, 0) = (2 * zNear) / (right - left);
            m.at(1, 1) = (2 * zNear) / (top - bottom);
            m.at(0, 2) = (right + left) / (right - left);
            m.at(1, 2) = (top + bottom) / (top - bottom);
            m.at(2, 2) = -(zFar + zNear) / (zFar - zNear);
            m.at(2, 3) = -(2 * zFar * zNear) / (zFar - zNear);
            m.at(3, 2) = -1;
            return m;
        }

        static Mat4x4<T> lookAt(const std::array<T, 3>& eye, const std::array<T, 3>& center, const std::array<T, 3>& up) {
            std::array<T, 3> f = {
                center[0] - eye[0],
                center[1] - eye[1],
                center[2] - eye[2]
            };
            T flen = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
            for (auto& v : f) v /= flen;

            std::array<T, 3> s = {
                f[1] * up[2] - f[2] * up[1],
                f[2] * up[0] - f[0] * up[2],
                f[0] * up[1] - f[1] * up[0]
            };
            T slen = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
            for (auto& v : s) v /= slen;

            std::array<T, 3> u = {
                s[1] * f[2] - s[2] * f[1],
                s[2] * f[0] - s[0] * f[2],
                s[0] * f[1] - s[1] * f[0]
            };

            Mat4x4<T> m = identity();
            m.at(0, 0) = s[0]; m.at(0, 1) = s[1]; m.at(0, 2) = s[2];
            m.at(1, 0) = u[0]; m.at(1, 1) = u[1]; m.at(1, 2) = u[2];
            m.at(2, 0) = -f[0]; m.at(2, 1) = -f[1]; m.at(2, 2) = -f[2];
            m = m * translate(-eye[0], -eye[1], -eye[2]);
            return m;
        }
    };

} // namespace math::scalar