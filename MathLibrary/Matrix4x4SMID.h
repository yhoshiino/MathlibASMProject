#pragma once
#include <immintrin.h>
#include <cmath>
#include <array>
#include <algorithm>
#include <cassert>

namespace math {

    struct alignas(16) Mat4x4f {
        union {
            __m128 col[4];     // Accès vectoriel SIMD par colonne
            float data[16];    // Accès plat column-major (compatible OpenGL/DirectX)
        };

        // Structure représentant un Quaternion de rotation
        struct Quaternion {
            float x = 0.0f, y = 0.0f, z = 0.0f, w = 1.0f;
        };

        // --- Constructeurs ---
        Mat4x4f() {
            col[0] = _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f);
            col[1] = _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f);
            col[2] = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
            col[3] = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
        }

        // --- Accès aux éléments (Column-Major) ---
        float& at(int row, int colIdx) {
            return data[colIdx * 4 + row];
        }

        const float& at(int row, int colIdx) const {
            return data[colIdx * 4 + row];
        }

        bool isIdentity() const {
            return at(0, 0) == 1.0f && at(0, 1) == 0.0f && at(0, 2) == 0.0f && at(0, 3) == 0.0f &&
                at(1, 0) == 0.0f && at(1, 1) == 1.0f && at(1, 2) == 0.0f && at(1, 3) == 0.0f &&
                at(2, 0) == 0.0f && at(2, 1) == 0.0f && at(2, 2) == 1.0f && at(2, 3) == 0.0f &&
                at(3, 0) == 0.0f && at(3, 1) == 0.0f && at(3, 2) == 0.0f && at(3, 3) == 1.0f;
        }

        // --- Matrices Spéciales ---
        static Mat4x4f identity() { return Mat4x4f(); }

        static Mat4x4f zero() {
            Mat4x4f m;
            m.col[0] = _mm_setzero_ps();
            m.col[1] = _mm_setzero_ps();
            m.col[2] = _mm_setzero_ps();
            m.col[3] = _mm_setzero_ps();
            return m;
        }

        // --- Transformations de base ---
        static Mat4x4f translate(float tx, float ty, float tz) {
            Mat4x4f m;
            m.col[3] = _mm_setr_ps(tx, ty, tz, 1.0f);
            return m;
        }

        static Mat4x4f scale(float sx, float sy, float sz) {
            Mat4x4f m;
            m.col[0] = _mm_setr_ps(sx, 0.0f, 0.0f, 0.0f);
            m.col[1] = _mm_setr_ps(0.0f, sy, 0.0f, 0.0f);
            m.col[2] = _mm_setr_ps(0.0f, 0.0f, sz, 0.0f);
            return m;
        }

        static Mat4x4f rotationX(float angleRad) {
            Mat4x4f m;
            float c = std::cos(angleRad);
            float s = std::sin(angleRad);
            m.col[1] = _mm_setr_ps(0.0f, c, s, 0.0f);
            m.col[2] = _mm_setr_ps(0.0f, -s, c, 0.0f);
            return m;
        }

        static Mat4x4f rotationY(float angleRad) {
            Mat4x4f m;
            float c = std::cos(angleRad);
            float s = std::sin(angleRad);
            m.col[0] = _mm_setr_ps(c, 0.0f, -s, 0.0f);
            m.col[2] = _mm_setr_ps(s, 0.0f, c, 0.0f);
            return m;
        }

        static Mat4x4f rotationZ(float angleRad) {
            Mat4x4f m;
            float c = std::cos(angleRad);
            float s = std::sin(angleRad);
            m.col[0] = _mm_setr_ps(c, s, 0.0f, 0.0f);
            m.col[1] = _mm_setr_ps(-s, c, 0.0f, 0.0f);
            return m;
        }

        static Mat4x4f trs(float tx, float ty, float tz, float rx, float ry, float rz, float sx, float sy, float sz) {
            return translate(tx, ty, tz) * rotationZ(rz) * rotationY(ry) * rotationX(rx) * scale(sx, sy, sz);
        }

        void setTRS(float tx, float ty, float tz, float rx, float ry, float rz, float sx, float sy, float sz) {
            *this = trs(tx, ty, tz, rx, ry, rz, sx, sy, sz);
        }

        // --- Multiplications de Vecteurs ---
        __m128 multiply(__m128 v) const {
            __m128 x = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
            __m128 y = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
            __m128 z = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
            __m128 w = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));

            __m128 r = _mm_mul_ps(col[0], x);
            r = _mm_add_ps(r, _mm_mul_ps(col[1], y));
            r = _mm_add_ps(r, _mm_mul_ps(col[2], z));
            r = _mm_add_ps(r, _mm_mul_ps(col[3], w));
            return r;
        }

        std::array<float, 3> multiplyPoint3x4(const std::array<float, 3>& p) const {
            __m128 v = _mm_setr_ps(p[0], p[1], p[2], 1.0f);
            __m128 r = multiply(v);
            alignas(16) float res[4];
            _mm_store_ps(res, r);
            return { res[0], res[1], res[2] };
        }

        std::array<float, 4> multiplyPoint(const std::array<float, 4>& p) const {
            __m128 v = _mm_setr_ps(p[0], p[1], p[2], p[3]);
            __m128 r = multiply(v);
            alignas(16) float res[4];
            _mm_store_ps(res, r);
            return { res[0], res[1], res[2], res[3] };
        }

        std::array<float, 3> multiplyVector(const std::array<float, 3>& v) const {
            __m128 vec = _mm_setr_ps(v[0], v[1], v[2], 0.0f);
            __m128 r = multiply(vec);
            alignas(16) float res[4];
            _mm_store_ps(res, r);
            return { res[0], res[1], res[2] };
        }

        // --- Extraction de composants ---
        std::array<float, 3> getPosition() const {
            return { at(0, 3), at(1, 3), at(2, 3) };
        }

        std::array<float, 3> lossyScale() const {
            float sx = std::sqrt(at(0, 0) * at(0, 0) + at(1, 0) * at(1, 0) + at(2, 0) * at(2, 0));
            float sy = std::sqrt(at(0, 1) * at(0, 1) + at(1, 1) * at(1, 1) + at(2, 1) * at(2, 1));
            float sz = std::sqrt(at(0, 2) * at(0, 2) + at(1, 2) * at(1, 2) + at(2, 2) * at(2, 2));
            return { sx, sy, sz };
        }

        // --- Opérateurs de Matrices ---
        Mat4x4f operator*(const Mat4x4f& o) const {
            Mat4x4f r;
            for (int j = 0; j < 4; ++j)
                r.col[j] = multiply(o.col[j]);
            return r;
        }

        Mat4x4f transpose() const {
            __m128 t0 = _mm_unpacklo_ps(col[0], col[1]);
            __m128 t1 = _mm_unpacklo_ps(col[2], col[3]);
            __m128 t2 = _mm_unpackhi_ps(col[0], col[1]);
            __m128 t3 = _mm_unpackhi_ps(col[2], col[3]);

            Mat4x4f r;
            r.col[0] = _mm_movelh_ps(t0, t1);
            r.col[1] = _mm_movehl_ps(t1, t0);
            r.col[2] = _mm_movelh_ps(t2, t3);
            r.col[3] = _mm_movehl_ps(t3, t2);
            return r;
        }

        // Lignes et colonnes
        std::array<float, 4> getColumn(int c) const {
            return { at(0, c), at(1, c), at(2, c), at(3, c) };
        }

        std::array<float, 4> getRow(int r) const {
            return { at(r, 0), at(r, 1), at(r, 2), at(r, 3) };
        }

        void setColumn(int c, const std::array<float, 4>& v) {
            col[c] = _mm_setr_ps(v[0], v[1], v[2], v[3]);
        }

        void setRow(int r, const std::array<float, 4>& v) {
            at(r, 0) = v[0]; at(r, 1) = v[1]; at(r, 2) = v[2]; at(r, 3) = v[3];
        }

        // --- Déterminant et Inversion ---
        float determinant() const {
            float s0 = at(0, 0) * at(1, 1) - at(1, 0) * at(0, 1);
            float s1 = at(0, 0) * at(1, 2) - at(1, 0) * at(0, 2);
            float s2 = at(0, 0) * at(1, 3) - at(1, 0) * at(0, 3);
            float s3 = at(0, 1) * at(1, 2) - at(1, 1) * at(0, 2);
            float s4 = at(0, 1) * at(1, 3) - at(1, 1) * at(0, 3);
            float s5 = at(0, 2) * at(1, 3) - at(1, 2) * at(0, 3);

            float c5 = at(2, 2) * at(3, 3) - at(3, 2) * at(2, 3);
            float c4 = at(2, 1) * at(3, 3) - at(3, 1) * at(2, 3);
            float c3 = at(2, 1) * at(3, 2) - at(3, 1) * at(2, 2);
            float c2 = at(2, 0) * at(3, 3) - at(3, 0) * at(2, 3);
            float c1 = at(2, 0) * at(3, 2) - at(3, 0) * at(2, 2);
            float c0 = at(2, 0) * at(3, 1) - at(3, 0) * at(2, 1);

            return s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        }

        bool validTRS() const {
            float det = determinant();
            return !std::isnan(det) && !std::isinf(det) && std::abs(det) > 1e-6f &&
                at(3, 0) == 0.0f && at(3, 1) == 0.0f && at(3, 2) == 0.0f && at(3, 3) == 1.0f;
        }

        Mat4x4f inverse() const {
            float det = determinant();
            if (std::abs(det) < 1e-8f) return zero();

            float invDet = 1.0f / det;
            Mat4x4f inv;

            inv.at(0, 0) = (at(1, 1) * (at(2, 2) * at(3, 3) - at(2, 3) * at(3, 2)) - at(1, 2) * (at(2, 1) * at(3, 3) - at(2, 3) * at(3, 1)) + at(1, 3) * (at(2, 1) * at(3, 2) - at(2, 2) * at(3, 1))) * invDet;
            inv.at(0, 1) = -(at(0, 1) * (at(2, 2) * at(3, 3) - at(2, 3) * at(3, 2)) - at(0, 2) * (at(2, 1) * at(3, 3) - at(2, 3) * at(3, 1)) + at(0, 3) * (at(2, 1) * at(3, 2) - at(2, 2) * at(3, 1))) * invDet;
            inv.at(0, 2) = (at(0, 1) * (at(1, 2) * at(3, 3) - at(1, 3) * at(3, 2)) - at(0, 2) * (at(1, 1) * at(3, 3) - at(1, 3) * at(3, 1)) + at(0, 3) * (at(1, 1) * at(3, 2) - at(1, 2) * at(3, 1))) * invDet;
            inv.at(0, 3) = -(at(0, 1) * (at(1, 2) * at(2, 3) - at(1, 3) * at(2, 2)) - at(0, 2) * (at(1, 1) * at(2, 3) - at(1, 3) * at(2, 1)) + at(0, 3) * (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1))) * invDet;

            inv.at(1, 0) = -(at(1, 0) * (at(2, 2) * at(3, 3) - at(2, 3) * at(3, 2)) - at(1, 2) * (at(2, 0) * at(3, 3) - at(2, 3) * at(3, 0)) + at(1, 3) * (at(2, 0) * at(3, 2) - at(2, 2) * at(3, 0))) * invDet;
            inv.at(1, 1) = (at(0, 0) * (at(2, 2) * at(3, 3) - at(2, 3) * at(3, 2)) - at(0, 2) * (at(2, 0) * at(3, 3) - at(2, 3) * at(3, 0)) + at(0, 3) * (at(2, 0) * at(3, 2) - at(2, 2) * at(3, 0))) * invDet;
            inv.at(1, 2) = -(at(0, 0) * (at(1, 2) * at(3, 3) - at(1, 3) * at(3, 2)) - at(0, 2) * (at(1, 0) * at(3, 3) - at(1, 3) * at(3, 0)) + at(0, 3) * (at(1, 0) * at(3, 2) - at(1, 2) * at(3, 0))) * invDet;
            inv.at(1, 3) = (at(0, 0) * (at(1, 2) * at(2, 3) - at(1, 3) * at(2, 2)) - at(0, 2) * (at(1, 0) * at(2, 3) - at(1, 3) * at(2, 0)) + at(0, 3) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0))) * invDet;

            inv.at(2, 0) = (at(1, 0) * (at(2, 1) * at(3, 3) - at(2, 3) * at(3, 1)) - at(1, 1) * (at(2, 0) * at(3, 3) - at(2, 3) * at(3, 0)) + at(1, 3) * (at(2, 0) * at(3, 1) - at(2, 1) * at(3, 0))) * invDet;
            inv.at(2, 1) = -(at(0, 0) * (at(2, 1) * at(3, 3) - at(2, 3) * at(3, 1)) - at(0, 1) * (at(2, 0) * at(3, 3) - at(2, 3) * at(3, 0)) + at(0, 3) * (at(2, 0) * at(3, 1) - at(2, 1) * at(3, 0))) * invDet;
            inv.at(2, 2) = (at(0, 0) * (at(1, 1) * at(3, 3) - at(1, 3) * at(3, 1)) - at(0, 1) * (at(1, 0) * at(3, 3) - at(1, 3) * at(3, 0)) + at(0, 3) * (at(1, 0) * at(3, 1) - at(1, 1) * at(3, 0))) * invDet;
            inv.at(2, 3) = -(at(0, 0) * (at(1, 1) * at(2, 3) - at(1, 3) * at(2, 1)) - at(0, 1) * (at(1, 0) * at(2, 3) - at(1, 3) * at(2, 0)) + at(0, 3) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0))) * invDet;

            inv.at(3, 0) = -(at(1, 0) * (at(2, 1) * at(3, 2) - at(2, 2) * at(3, 1)) - at(1, 1) * (at(2, 0) * at(3, 2) - at(2, 2) * at(3, 0)) + at(1, 2) * (at(2, 0) * at(3, 1) - at(2, 1) * at(3, 0))) * invDet;
            inv.at(3, 1) = (at(0, 0) * (at(2, 1) * at(3, 2) - at(2, 2) * at(3, 1)) - at(0, 1) * (at(2, 0) * at(3, 2) - at(2, 2) * at(3, 0)) + at(0, 2) * (at(2, 0) * at(3, 1) - at(2, 1) * at(3, 0))) * invDet;
            inv.at(3, 2) = -(at(0, 0) * (at(1, 1) * at(3, 2) - at(1, 2) * at(3, 1)) - at(0, 1) * (at(1, 0) * at(3, 2) - at(1, 2) * at(3, 0)) + at(0, 2) * (at(1, 0) * at(3, 1) - at(1, 1) * at(3, 0))) * invDet;
            inv.at(3, 3) = (at(0, 0) * (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1)) - at(0, 1) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0)) + at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0))) * invDet;

            return inv;
        }

        // --- Projections (Rendu 3D) ---
        static Mat4x4f perspective(float fovRad, float aspect, float nearPlane, float farPlane) {
            float f = 1.0f / std::tan(fovRad / 2.0f);
            Mat4x4f m = zero();
            m.at(0, 0) = f / aspect;
            m.at(1, 1) = f;
            m.at(2, 2) = (farPlane + nearPlane) / (nearPlane - farPlane);
            m.at(2, 3) = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
            m.at(3, 2) = -1.0f;
            return m;
        }

        static Mat4x4f frustum(float left, float right, float bottom, float top, float nearPlane, float farPlane) {
            Mat4x4f m = zero();
            m.at(0, 0) = (2.0f * nearPlane) / (right - left);
            m.at(1, 1) = (2.0f * nearPlane) / (top - bottom);
            m.at(0, 2) = (right + left) / (right - left);
            m.at(1, 2) = (top + bottom) / (top - bottom);
            m.at(2, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
            m.at(2, 3) = -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane);
            m.at(3, 2) = -1.0f;
            return m;
        }

        static Mat4x4f ortho(float left, float right, float bottom, float top, float nearPlane, float farPlane) {
            Mat4x4f m;
            m.at(0, 0) = 2.0f / (right - left);
            m.at(1, 1) = 2.0f / (top - bottom);
            m.at(2, 2) = -2.0f / (farPlane - nearPlane);
            m.at(0, 3) = -(right + left) / (right - left);
            m.at(1, 3) = -(top + bottom) / (top - bottom);
            m.at(2, 3) = -(farPlane + nearPlane) / (farPlane - nearPlane);
            return m;
        }

        static Mat4x4f lookAt(const std::array<float, 3>& eye, const std::array<float, 3>& center, const std::array<float, 3>& up) {
            float fx = center[0] - eye[0], fy = center[1] - eye[1], fz = center[2] - eye[2];
            float flen = std::sqrt(fx * fx + fy * fy + fz * fz);
            fx /= flen; fy /= flen; fz /= flen;

            // cross(f, up)
            float sx = fy * up[2] - fz * up[1];
            float sy = fz * up[0] - fx * up[2];
            float sz = fx * up[1] - fy * up[0];
            float slen = std::sqrt(sx * sx + sy * sy + sz * sz);
            sx /= slen; sy /= slen; sz /= slen;

            // cross(s, f)
            float ux = sy * fz - sz * fy;
            float uy = sz * fx - sx * fz;
            float uz = sx * fy - sy * fx;

            Mat4x4f m;
            m.at(0, 0) = sx;  m.at(0, 1) = sy;  m.at(0, 2) = sz;  m.at(0, 3) = -(sx * eye[0] + sy * eye[1] + sz * eye[2]);
            m.at(1, 0) = ux;  m.at(1, 1) = uy;  m.at(1, 2) = uz;  m.at(1, 3) = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
            m.at(2, 0) = -fx; m.at(2, 1) = -fy; m.at(2, 2) = -fz; m.at(2, 3) = (fx * eye[0] + fy * eye[1] + fz * eye[2]);
            m.at(3, 0) = 0.0f; m.at(3, 1) = 0.0f; m.at(3, 2) = 0.0f; m.at(3, 3) = 1.0f;
            return m;
        }

        // --- Conversion vers Quaternion ---
        Quaternion rotation() const {
            auto ls = lossyScale();
            float m00 = at(0, 0) / ls[0], m01 = at(0, 1) / ls[1], m02 = at(0, 2) / ls[2];
            float m10 = at(1, 0) / ls[0], m11 = at(1, 1) / ls[1], m12 = at(1, 2) / ls[2];
            float m20 = at(2, 0) / ls[0], m21 = at(2, 1) / ls[1], m22 = at(2, 2) / ls[2];

            float trace = m00 + m11 + m22;
            Quaternion q;

            if (trace > 0.0f) {
                float s = 0.5f / std::sqrt(trace + 1.0f);
                q.w = 0.25f / s;
                q.x = (m21 - m12) * s;
                q.y = (m02 - m20) * s;
                q.z = (m10 - m01) * s;
            }
            else if (m00 > m11 && m00 > m22) {
                float s = 2.0f * std::sqrt(1.0f + m00 - m11 - m22);
                q.w = (m21 - m12) / s;
                q.x = 0.25f * s;
                q.y = (m01 + m10) / s;
                q.z = (m02 + m20) / s;
            }
            else if (m11 > m22) {
                float s = 2.0f * std::sqrt(1.0f + m11 - m00 - m22);
                q.w = (m02 - m20) / s;
                q.x = (m01 + m10) / s;
                q.y = 0.25f * s;
                q.z = (m12 + m21) / s;
            }
            else {
                float s = 2.0f * std::sqrt(1.0f + m22 - m00 - m11);
                q.w = (m10 - m01) / s;
                q.x = (m02 + m20) / s;
                q.y = (m12 + m21) / s;
                q.z = 0.25f * s;
            }
            return q;
        }
    };

} // namespace math