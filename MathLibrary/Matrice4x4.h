#pragma once
#include <array>
#include <cmath>

namespace math::scalar {

    // Convention : stockage column-major data[col*4 + row], vecteurs colonnes (M * v),
    // translation dans la colonne 3, angles en radians, repère main droite.
    // Composition : A * B applique B en premier, puis A.
    template<typename T = float>
    class Mat4x4 {
    public:
        std::array<T, 16> data;

        // Identité
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

        static Mat4x4<T> translate(T tx, T ty, T tz) {
            Mat4x4<T> m;
            m.at(0, 3) = tx;
            m.at(1, 3) = ty;
            m.at(2, 3) = tz;
            return m;
        }

        static Mat4x4<T> rotationZ(T angle) {
            T c = std::cos(angle);
            T s = std::sin(angle);
            Mat4x4<T> m;
            m.at(0, 0) = c;
            m.at(0, 1) = -s;
            m.at(1, 0) = s;
            m.at(1, 1) = c;
            return m;
        }

        Mat4x4<T> operator*(const Mat4x4<T>& other) const {
            Mat4x4<T> result;
            result.data.fill(0);
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    for (int k = 0; k < 4; ++k) {
                        result.at(row, col) += at(row, k) * other.at(k, col);
                    }
                }
            }
            return result;
        }
    };

} // namespace math::scalar