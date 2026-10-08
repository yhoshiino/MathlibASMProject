#pragma once
#include <cmath>

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

        T dot(const Vector3& o) const { return x * o.x + y * o.y + z * o.z; }

        T magnitude() const { return std::sqrt(dot(*this)); }

        Vector3 normalized() const {
            const T len2 = dot(*this);
            // Écrit sous cette forme pour que NaN => vecteur nul (comme cmpgt côté SIMD)
            if (!(len2 > static_cast<T>(kNormalizeEpsSq))) return Vector3(0, 0, 0);
            const T len = std::sqrt(len2);
            return Vector3(x / len, y / len, z / len);
        }
    };

} // namespace math::scalar