    #pragma once
    #include <array>
    #include <cmath>
    #include <iostream>
    #include "Vector2.h"

    namespace math {

        template<typename T = float>
        class Mat3x3 {
        public:
            // Column-major storage: 3 columns of 3 elements each
            std::array<T, 9> data;

            // Default constructor: identity matrix
            Mat3x3() {
                data = {
                    1, 0, 0,  // column 0
                    0, 1, 0,  // column 1
                    0, 0, 1   // column 2
                };
            }

            // Access element by (row, column)
            T& at(int row, int col) {
                return data[col * 3 + row];
            }

            const T& at(int row, int col) const {
                return data[col * 3 + row];
            }

            // Matrix Identity
            static Mat3x3<T> identity() {
                return Mat3x3<T>();
            }

            // Matrix × homogeneous vector multiplication (x, y, 1)
            std::array<T, 3> multiply(const std::array<T, 3>& vec) const {
                std::array<T, 3> result = { 0, 0, 0 };
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        result[row] += at(row, col) * vec[col];
                    }
                }
                return result;
            }

            // Matrix × matrix multiplication
            Mat3x3 operator*(const Mat3x3& other) const {
                Mat3x3 result;
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        result.at(row, col) = 0;
                        for (int k = 0; k < 3; ++k) {
                            result.at(row, col) += at(row, k) * other.at(k, col);
                        }
                    }
                }
                return result;
            }

            // Create a rotation matrix (angle in radians)
            static Mat3x3 rotation(T angle) {
                T c = std::cos(angle);
                T s = std::sin(angle);
                Mat3x3 rot;
                rot.at(0, 0) = c;  rot.at(0, 1) = -s;
                rot.at(1, 0) = s;  rot.at(1, 1) = c;
                rot.at(2, 2) = 1;
                return rot;
            }

            // Create a scaling matrix
            static Mat3x3 scale(T sx, T sy) {
                Mat3x3 m;
                m.at(0, 0) = sx;
                m.at(1, 1) = sy;
                m.at(2, 2) = 1;
                return m;
            }

            // Create a translation matrix
            static Mat3x3 translate(T tx, T ty) {
                Mat3x3 m;
                m.at(0, 0) = 1;
                m.at(1, 1) = 1;
                m.at(2, 2) = 1;
                m.at(0, 2) = tx;
                m.at(1, 2) = ty;
                return m;
            }

            // Print matrix to console
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

        template<typename T>
        Vector2<T> operator*(const Mat3x3<T>& m, const Vector2<T>& v) {
            return {
                m.at(0, 0) * v.x + m.at(0, 1) * v.y + m.at(0, 2),
                m.at(1, 0) * v.x + m.at(1, 1) * v.y + m.at(1, 2)
            };
        }
    } // namespace math