#include "pch.h"
#include "CppUnitTest.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <immintrin.h>
#include "../MathLibrary/Matrix4x4SIMD.h" // Vérifiez le chemin vers votre fichier Mat4x4f.h

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace math;

namespace
{
	using Mat = Mat4x4f;
	using V3 = std::array<float, 3>;
	using V4 = std::array<float, 4>;

	constexpr float EPS = 1e-4f;
	constexpr float PI = 3.14159265358979f;

	// Comparaison de deux tableaux V3
	void AssertArrayNear(const V3& expected, const V3& actual, float eps = EPS)
	{
		for (size_t i = 0; i < 3; ++i)
		{
			std::wstring msg = L"index " + std::to_wstring(i);
			Assert::AreEqual(expected[i], actual[i], eps, msg.c_str());
		}
	}

	// Comparaison de deux tableaux V4
	void AssertArrayNear(const V4& expected, const V4& actual, float eps = EPS)
	{
		for (size_t i = 0; i < 4; ++i)
		{
			std::wstring msg = L"index " + std::to_wstring(i);
			Assert::AreEqual(expected[i], actual[i], eps, msg.c_str());
		}
	}

	// Comparaison de deux matrices Mat4x4f
	void AssertMatNear(const Mat& expected, const Mat& actual, float eps = EPS)
	{
		for (int row = 0; row < 4; ++row)
		{
			for (int col = 0; col < 4; ++col)
			{
				std::wstring msg = L"at(" + std::to_wstring(row) + L", " + std::to_wstring(col) + L")";
				Assert::AreEqual(expected.at(row, col), actual.at(row, col), eps, msg.c_str());
			}
		}
	}

	// Matrice remplie séquentiellement selon la disposition Column-Major
	Mat MakeSequential(float offset)
	{
		Mat m;
		for (int row = 0; row < 4; ++row)
		{
			for (int col = 0; col < 4; ++col)
			{
				// col * 4 + row garantit que at(1, 2) aura bien la valeur 9 + offset
				m.at(row, col) = static_cast<float>(col * 4 + row) + offset;
			}
		}
		return m;
	}

	// Multiplication scalaire de référence (triple boucle)
	Mat ReferenceMultiply(const Mat& a, const Mat& b)
	{
		Mat r = Mat::zero();
		for (int row = 0; row < 4; ++row)
		{
			for (int col = 0; col < 4; ++col)
			{
				float sum = 0.0f;
				for (int k = 0; k < 4; ++k)
					sum += a.at(row, k) * b.at(k, col);
				r.at(row, col) = sum;
			}
		}
		return r;
	}
}

namespace Mat4x4fTests
{
	TEST_CLASS(Mat4x4fSIMDTests)
	{
	public:

		// --- 1. CONSTRUCTEURS, ACCÈS ET ALIGNEMENT SIMD ---
		TEST_METHOD(ConstructorsAccessAndAlignment)
		{
			// Constructeur par défaut = identité
			Mat m;
			Assert::IsTrue(m.isIdentity());
			AssertMatNear(m, Mat::identity());

			// Matrice Nulle
			Mat z = Mat::zero();
			for (int r = 0; r < 4; ++r)
				for (int c = 0; c < 4; ++c)
					Assert::AreEqual(0.0f, z.at(r, c));

			// Alignement mémoire obligatoire pour SIMD (16 octets)
			Assert::AreEqual(static_cast<std::uintptr_t>(0), reinterpret_cast<std::uintptr_t>(&m) % 16);
			Assert::AreEqual(static_cast<std::uintptr_t>(0), reinterpret_cast<std::uintptr_t>(m.data) % 16);
			Assert::AreEqual(static_cast<size_t>(16 * sizeof(float)), sizeof(Mat));

			// Disposition column-major : at(row, col) == data[col * 4 + row]
			Mat seq = MakeSequential(0.0f);
			Assert::AreEqual(9.0f, seq.at(1, 2));
			Assert::AreEqual(9.0f, seq.data[2 * 4 + 1]);
		}

		// --- 2. ACCÈS PAR REGISTRES __m128 ET COLONNES/LIGNES ---
		TEST_METHOD(ColumnAndRowAccess)
		{
			Mat m = Mat::identity();

			// setColumn
			V4 newCol2 = { 1.0f, 2.0f, 3.0f, 4.0f };
			m.setColumn(2, newCol2);
			AssertArrayNear(newCol2, m.getColumn(2));

			// Validation du registre __m128 sous-jacent
			alignas(16) float store[4];
			_mm_store_ps(store, m.col[2]);
			Assert::AreEqual(1.0f, store[0]);
			Assert::AreEqual(2.0f, store[1]);
			Assert::AreEqual(3.0f, store[2]);
			Assert::AreEqual(4.0f, store[3]);

			// setRow
			V4 newRow1 = { 5.0f, 6.0f, 7.0f, 8.0f };
			m.setRow(1, newRow1);
			AssertArrayNear(newRow1, m.getRow(1));
		}

		// --- 3. TRANSFORMATIONS GEOMETRIQUES (TRS) ---
		TEST_METHOD(TranslationAndScaling)
		{
			// Translation
			Mat t = Mat::translate(10.0f, -20.0f, 30.0f);
			AssertArrayNear(V3{ 10.0f, -20.0f, 30.0f }, t.getPosition());

			// Scale
			Mat s = Mat::scale(2.0f, 3.0f, 4.0f);
			AssertArrayNear(V3{ 2.0f, 3.0f, 4.0f }, s.lossyScale());

			// Composition TRS
			Mat trsMat = Mat::trs(10.0f, -20.0f, 30.0f, 0.0f, 0.0f, 0.0f, 2.0f, 3.0f, 4.0f);
			AssertMatNear(t * s, trsMat);
			Assert::IsTrue(trsMat.validTRS());
		}

		TEST_METHOD(Rotations)
		{
			float angle = PI / 2.0f; // 90°

			// Rotation X : Y -> Z
			Mat rx = Mat::rotationX(angle);
			AssertArrayNear(V3{ 0.0f, 0.0f, 1.0f }, rx.multiplyVector(V3{ 0.0f, 1.0f, 0.0f }));

			// Rotation Y : Z -> X
			Mat ry = Mat::rotationY(angle);
			AssertArrayNear(V3{ 1.0f, 0.0f, 0.0f }, ry.multiplyVector(V3{ 0.0f, 0.0f, 1.0f }));

			// Rotation Z : X -> Y
			Mat rz = Mat::rotationZ(angle);
			AssertArrayNear(V3{ 0.0f, 1.0f, 0.0f }, rz.multiplyVector(V3{ 1.0f, 0.0f, 0.0f }));
		}

		// --- 4. MULTIPLICATIONS ET SIMD ---
		TEST_METHOD(MatrixMultiplicationSIMD)
		{
			Mat a = MakeSequential(1.0f);
			Mat b = MakeSequential(-4.0f);

			// Comparaison entre l'opérateur * (SIMD) et le calcul scalaire de référence
			AssertMatNear(ReferenceMultiply(a, b), a * b);

			// Propriété de l'élément neutre
			AssertMatNear(a, a * Mat::identity());
			AssertMatNear(a, Mat::identity() * a);
		}

		TEST_METHOD(VectorAndPointMultiplication)
		{
			Mat t = Mat::translate(5.0f, 10.0f, 15.0f);

			// Vecteur directionnel (W = 0, insensible à la translation)
			V3 dir = { 1.0f, 2.0f, 3.0f };
			AssertArrayNear(dir, t.multiplyVector(dir));

			// Point 3D (homogène W = 1, subit la translation)
			AssertArrayNear(V3{ 6.0f, 12.0f, 18.0f }, t.multiplyPoint3x4(dir));

			// Point 4D explicite
			V4 pt4 = { 1.0f, 2.0f, 3.0f, 1.0f };
			AssertArrayNear(V4{ 6.0f, 12.0f, 18.0f, 1.0f }, t.multiplyPoint(pt4));

			// Multiplier directement un registre __m128
			alignas(16) float store[4];
			_mm_store_ps(store, t.multiply(_mm_setr_ps(1.0f, 2.0f, 3.0f, 1.0f)));
			Assert::AreEqual(6.0f, store[0], EPS);
			Assert::AreEqual(12.0f, store[1], EPS);
			Assert::AreEqual(18.0f, store[2], EPS);
			Assert::AreEqual(1.0f, store[3], EPS);
		}

		// --- 5. OPERATEURS ALGEBRIQUES (TRANSPOSITION, DETERMINANT, INVERSION) ---
		TEST_METHOD(Transpose)
		{
			Mat m = Mat::translate(1.0f, 2.0f, 3.0f);
			Mat transposed = m.transpose();

			Assert::AreEqual(1.0f, transposed.at(3, 0));
			Assert::AreEqual(2.0f, transposed.at(3, 1));
			Assert::AreEqual(3.0f, transposed.at(3, 2));
		}

		TEST_METHOD(Determinant)
		{
			Assert::AreEqual(1.0f, Mat::identity().determinant(), EPS);

			Mat s = Mat::scale(2.0f, 3.0f, 4.0f);
			Assert::AreEqual(24.0f, s.determinant(), EPS);
		}

		TEST_METHOD(Inverse)
		{
			Mat m = Mat::trs(2.0f, -1.0f, 0.5f, 0.1f, 0.2f, 0.3f, 1.5f, 0.8f, 2.0f);
			Mat inv = m.inverse();

			// M * M^-1 == Identité
			AssertMatNear(Mat::identity(), m * inv);

			// Matrice non inversible -> Matrice nulle
			Mat zeroMat = Mat::zero();
			AssertMatNear(zeroMat, zeroMat.inverse());
		}

		// --- 6. PROJECTIONS ET CAMERA ---
		TEST_METHOD(LookAt)
		{
			V3 eye = { 0.0f, 0.0f, 5.0f };
			V3 center = { 0.0f, 0.0f, 0.0f };
			V3 up = { 0.0f, 1.0f, 0.0f };

			Mat view = Mat::lookAt(eye, center, up);

			// La position de l'œil dans l'espace vue doit devenir l'origine (0,0,0)
			AssertArrayNear(V3{ 0.0f, 0.0f, 0.0f }, view.multiplyPoint3x4(eye));
		}

		TEST_METHOD(Projections)
		{
			// Perspective
			Mat p = Mat::perspective(PI / 3.0f, 16.0f / 9.0f, 0.1f, 100.0f);
			Assert::AreEqual(-1.0f, p.at(3, 2));

			// Orthographique
			Mat o = Mat::ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);
			Assert::AreEqual(1.0f, o.at(3, 3));

			// Frustum
			Mat f = Mat::frustum(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);
			Assert::AreEqual(-1.0f, f.at(3, 2));
		}

		// --- 7. CONVERSION EN QUATERNION ---
		TEST_METHOD(RotationToQuaternion)
		{
			float angle = PI / 2.0f; // 90° autour de Z
			Mat rotZ = Mat::rotationZ(angle);

			Mat4x4f::Quaternion q = rotZ.rotation();

			// 90° autour de Z -> w = cos(45°) = 0.7071, z = sin(45°) = 0.7071
			Assert::AreEqual(std::cos(angle / 2.0f), q.w, EPS);
			Assert::AreEqual(0.0f, q.x, EPS);
			Assert::AreEqual(0.0f, q.y, EPS);
			Assert::AreEqual(std::sin(angle / 2.0f), q.z, EPS);
		}
	};
}