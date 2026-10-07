#include "pch.h"
#include "CppUnitTest.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include "../MathLibrary/Matrix3x3SMID.h" // Vérifie le chemin vers Mat3x3.h

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace math;

namespace
{
	using Mat = Mat3x3<float>;   // version SIMD (spécialisation)
	using MatD = Mat3x3<double>; // version scalaire générique, sert de référence
	using V3 = std::array<float, 3>;

	constexpr float EPS = 1e-4f;
	constexpr float PI = 3.14159265358979f;

	// Compare deux tableaux élément par élément
	void AssertArrayNear(const V3& expected, const V3& actual, float eps = EPS)
	{
		for (size_t i = 0; i < 3; ++i)
		{
			std::wstring msg = L"index " + std::to_wstring(i);
			Assert::AreEqual(expected[i], actual[i], eps, msg.c_str());
		}
	}

	// Compare deux matrices élément par élément
	void AssertMatNear(const Mat& expected, const Mat& actual, float eps = EPS)
	{
		for (int row = 0; row < 3; ++row)
		{
			for (int col = 0; col < 3; ++col)
			{
				std::wstring msg = L"at(" + std::to_wstring(row) + L", " + std::to_wstring(col) + L")";
				Assert::AreEqual(expected.at(row, col), actual.at(row, col), eps, msg.c_str());
			}
		}
	}

	// Vérifie que la 4e lane (padding) de chaque colonne vaut toujours 0
	void AssertPaddingIsZero(const Mat& m)
	{
		Assert::AreEqual(0.0f, m.data[3]);
		Assert::AreEqual(0.0f, m.data[7]);
		Assert::AreEqual(0.0f, m.data[11]);
	}

	// Matrice remplie avec at(row, col) = row * 3 + col + offset
	Mat MakeSequential(float offset)
	{
		Mat m;
		for (int row = 0; row < 3; ++row)
			for (int col = 0; col < 3; ++col)
				m.at(row, col) = static_cast<float>(row * 3 + col) + offset;
		return m;
	}

	// Multiplication de référence, en scalaire pur (triple boucle)
	Mat ReferenceMultiply(const Mat& a, const Mat& b)
	{
		Mat r;
		for (int row = 0; row < 3; ++row)
		{
			for (int col = 0; col < 3; ++col)
			{
				float sum = 0.0f;
				for (int k = 0; k < 3; ++k)
					sum += a.at(row, k) * b.at(k, col);
				r.at(row, col) = sum;
			}
		}
		return r;
	}
}

namespace Mat3x3Tests
{
	TEST_CLASS(Mat3x3Tests)
	{
	public:

		// --- 1. CONSTRUCTEURS, ACCÈS ET STOCKAGE ---
		TEST_METHOD(ConstructorsAndAccess)
		{
			// Constructeur par défaut = identité
			Mat m;
			for (int row = 0; row < 3; ++row)
				for (int col = 0; col < 3; ++col)
					Assert::AreEqual(row == col ? 1.0f : 0.0f, m.at(row, col));

			// identity() = constructeur par défaut
			AssertMatNear(m, Mat::identity());

			// Stockage column-major avec padding : at(row, col) == data[col * 4 + row]
			Mat s = MakeSequential(0.0f);
			Assert::AreEqual(5.0f, s.at(1, 2));
			Assert::AreEqual(5.0f, s.data[2 * 4 + 1]);
			Assert::AreEqual(s.at(2, 0), s.data[0 * 4 + 2]);

			// Version const de at()
			const Mat& cs = s;
			Assert::AreEqual(7.0f, cs.at(2, 1));

			// Le padding n'est jamais touché par at()
			AssertPaddingIsZero(s);
		}

		// --- 2. ALIGNEMENT MÉMOIRE (SIMD) ---
		TEST_METHOD(MemoryAlignment)
		{
			Mat m;
			Assert::AreEqual(static_cast<std::uintptr_t>(0), reinterpret_cast<std::uintptr_t>(&m) % 16);
			Assert::AreEqual(static_cast<std::uintptr_t>(0), reinterpret_cast<std::uintptr_t>(m.data) % 16);

			// 3 colonnes de 4 floats
			Assert::AreEqual(static_cast<size_t>(12 * sizeof(float)), sizeof(Mat));
		}

		// --- 3. ACCÈS AUX COLONNES (REGISTRES SIMD) ---
		TEST_METHOD(ColumnAccess)
		{
			Mat m;
			m.setColumn(1, _mm_setr_ps(7.0f, 8.0f, 9.0f, 0.0f));

			Assert::AreEqual(7.0f, m.at(0, 1));
			Assert::AreEqual(8.0f, m.at(1, 1));
			Assert::AreEqual(9.0f, m.at(2, 1));

			alignas(16) float out[4];
			_mm_store_ps(out, m.getColumn(1));
			Assert::AreEqual(7.0f, out[0]);
			Assert::AreEqual(8.0f, out[1]);
			Assert::AreEqual(9.0f, out[2]);
			Assert::AreEqual(0.0f, out[3]);

			// Les autres colonnes ne sont pas modifiées
			Assert::AreEqual(1.0f, m.at(0, 0));
			Assert::AreEqual(1.0f, m.at(2, 2));
		}

		// --- 4. TRANSLATION ---
		TEST_METHOD(Translation)
		{
			Mat t = Mat::translate(10.0f, 20.0f);

			// Disposition en mémoire : la translation est dans la colonne 2
			Assert::AreEqual(10.0f, t.data[8]);
			Assert::AreEqual(20.0f, t.data[9]);
			Assert::AreEqual(1.0f, t.data[10]);

			// Un point (x, y, 1) est déplacé
			AssertArrayNear(V3{ 11.0f, 22.0f, 1.0f }, t.multiply(V3{ 1.0f, 2.0f, 1.0f }));

			// Une direction (x, y, 0) n'est PAS déplacée
			AssertArrayNear(V3{ 1.0f, 2.0f, 0.0f }, t.multiply(V3{ 1.0f, 2.0f, 0.0f }));

			// Le reste de la matrice est l'identité
			Assert::AreEqual(1.0f, t.at(0, 0));
			Assert::AreEqual(1.0f, t.at(1, 1));
			Assert::AreEqual(0.0f, t.at(0, 1));
			Assert::AreEqual(0.0f, t.at(1, 0));
			AssertPaddingIsZero(t);
		}

		// --- 5. MISE À L'ÉCHELLE ---
		TEST_METHOD(Scaling)
		{
			Mat s = Mat::scale(2.0f, 3.0f);

			AssertArrayNear(V3{ 2.0f, 3.0f, 1.0f }, s.multiply(V3{ 1.0f, 1.0f, 1.0f }));
			AssertArrayNear(V3{ -4.0f, 9.0f, 1.0f }, s.multiply(V3{ -2.0f, 3.0f, 1.0f }));

			// Échelle (1, 1) = identité
			AssertMatNear(Mat::identity(), Mat::scale(1.0f, 1.0f));

			Assert::AreEqual(1.0f, s.at(2, 2));
			AssertPaddingIsZero(s);
		}

		// --- 6. ROTATION (radians, sens trigonométrique) ---
		TEST_METHOD(Rotation)
		{
			// 90° : X -> Y et Y -> -X
			Mat r90 = Mat::rotation(PI / 2);
			AssertArrayNear(V3{ 0.0f, 1.0f, 1.0f }, r90.multiply(V3{ 1.0f, 0.0f, 1.0f }));
			AssertArrayNear(V3{ -1.0f, 0.0f, 1.0f }, r90.multiply(V3{ 0.0f, 1.0f, 1.0f }));

			// 180° : X -> -X
			AssertArrayNear(V3{ -1.0f, 0.0f, 1.0f }, Mat::rotation(PI).multiply(V3{ 1.0f, 0.0f, 1.0f }));

			// 0° = identité
			AssertMatNear(Mat::identity(), Mat::rotation(0.0f));

			// Une rotation conserve la longueur : (3, 4) reste de longueur 5
			V3 r = Mat::rotation(0.7f).multiply(V3{ 3.0f, 4.0f, 1.0f });
			Assert::AreEqual(5.0f, std::sqrt(r[0] * r[0] + r[1] * r[1]), EPS);

			// rotation(a) * rotation(-a) = identité
			AssertMatNear(Mat::identity(), Mat::rotation(0.9f) * Mat::rotation(-0.9f));

			// rotation(a) * rotation(b) = rotation(a + b)
			AssertMatNear(Mat::rotation(1.3f), Mat::rotation(0.4f) * Mat::rotation(0.9f));

			// Structure : [c -s; s c] dans le bloc 2x2, 1 en (2, 2)
			Mat m = Mat::rotation(0.5f);
			Assert::AreEqual(std::cos(0.5f), m.at(0, 0), EPS);
			Assert::AreEqual(-std::sin(0.5f), m.at(0, 1), EPS);
			Assert::AreEqual(std::sin(0.5f), m.at(1, 0), EPS);
			Assert::AreEqual(std::cos(0.5f), m.at(1, 1), EPS);
			Assert::AreEqual(1.0f, m.at(2, 2));
			AssertPaddingIsZero(m);
		}

		// --- 7. MULTIPLICATION DE MATRICES ---
		TEST_METHOD(Multiplication)
		{
			Mat m = MakeSequential(1.0f);

			// Identité = élément neutre
			AssertMatNear(m, Mat::identity() * m);
			AssertMatNear(m, m * Mat::identity());

			// Comparaison avec une multiplication scalaire de référence
			Mat a = MakeSequential(1.0f);
			Mat b = MakeSequential(-4.0f);
			AssertMatNear(ReferenceMultiply(a, b), a * b);
			AssertMatNear(ReferenceMultiply(b, a), b * a);

			// Valeur connue : [1 2 3; 4 5 6; 7 8 9] * [1 0 0; 0 1 0; 0 0 1] = lui-même,
			// et sa 1re colonne multipliée par 2 via scale(2, 2)
			Mat sc = Mat::scale(2.0f, 2.0f);
			Mat prod = sc * a;
			Assert::AreEqual(2.0f, prod.at(0, 0), EPS); // 2 * 1
			Assert::AreEqual(4.0f, prod.at(0, 1), EPS); // 2 * 2
			Assert::AreEqual(8.0f, prod.at(1, 0), EPS); // 2 * 4
			Assert::AreEqual(7.0f, prod.at(2, 0), EPS); // la dernière ligne n'est pas modifiée

			// Ordre d'application : B est appliquée en premier.
			// Scale puis translate : (1,1) -> (2,2) -> (3,4)
			Mat scaleThenTranslate = Mat::translate(1.0f, 2.0f) * Mat::scale(2.0f, 2.0f);
			AssertArrayNear(V3{ 3.0f, 4.0f, 1.0f }, scaleThenTranslate.multiply(V3{ 1.0f, 1.0f, 1.0f }));

			// Translate puis scale : (1,1) -> (2,3) -> (4,6)
			Mat translateThenScale = Mat::scale(2.0f, 2.0f) * Mat::translate(1.0f, 2.0f);
			AssertArrayNear(V3{ 4.0f, 6.0f, 1.0f }, translateThenScale.multiply(V3{ 1.0f, 1.0f, 1.0f }));

			// La multiplication n'est pas commutative
			Assert::IsFalse(std::abs(scaleThenTranslate.at(0, 2) - translateThenScale.at(0, 2)) < EPS);

			// Associativité
			Mat r = Mat::rotation(0.3f);
			Mat t = Mat::translate(1.0f, 2.0f);
			Mat s = Mat::scale(2.0f, 3.0f);
			AssertMatNear((r * t) * s, r * (t * s));

			// Le padding reste à 0 après multiplication
			AssertPaddingIsZero(a * b);
			AssertPaddingIsZero((r * t) * s);
		}

		// --- 8. MULTIPLICATION MATRICE x VECTEUR (REGISTRE __m128) ---
		TEST_METHOD(MultiplyRawRegister)
		{
			Mat m = Mat::translate(10.0f, 20.0f);

			alignas(16) float out[4];
			_mm_store_ps(out, m.multiply(_mm_setr_ps(1.0f, 2.0f, 1.0f, 0.0f)));

			Assert::AreEqual(11.0f, out[0], EPS);
			Assert::AreEqual(22.0f, out[1], EPS);
			Assert::AreEqual(1.0f, out[2], EPS);
			Assert::AreEqual(0.0f, out[3], EPS); // la lane de padding reste à 0

			// La 4e lane du vecteur d'entrée est ignorée
			_mm_store_ps(out, m.multiply(_mm_setr_ps(1.0f, 2.0f, 1.0f, 999.0f)));
			Assert::AreEqual(11.0f, out[0], EPS);
			Assert::AreEqual(22.0f, out[1], EPS);
			Assert::AreEqual(1.0f, out[2], EPS);
			Assert::AreEqual(0.0f, out[3], EPS);
		}

		// --- 9. TRANSFORMATION D'UN VECTOR2 ---
		TEST_METHOD(Vector2Transform)
		{
			// Translation seule
			Vector2<float> p{ 1.0f, 2.0f };
			Vector2<float> t = Mat::translate(10.0f, 20.0f) * p;
			Assert::AreEqual(11.0f, t.x, EPS);
			Assert::AreEqual(22.0f, t.y, EPS);

			// Identité
			Vector2<float> i = Mat::identity() * p;
			Assert::AreEqual(1.0f, i.x, EPS);
			Assert::AreEqual(2.0f, i.y, EPS);

			// Scale x2 -> rotation 90° -> translation (5, 6) : (1,0) -> (2,0) -> (0,2) -> (5,8)
			Mat m = Mat::translate(5.0f, 6.0f) * Mat::rotation(PI / 2) * Mat::scale(2.0f, 2.0f);
			Vector2<float> r = m * Vector2<float>{ 1.0f, 0.0f };
			Assert::AreEqual(5.0f, r.x, EPS);
			Assert::AreEqual(8.0f, r.y, EPS);

			// Appliquer la matrice composée = appliquer les matrices une par une
			Vector2<float> step = Mat::scale(2.0f, 2.0f) * Vector2<float>{ 1.0f, 0.0f };
			step = Mat::rotation(PI / 2) * step;
			step = Mat::translate(5.0f, 6.0f) * step;
			Assert::AreEqual(r.x, step.x, EPS);
			Assert::AreEqual(r.y, step.y, EPS);
		}

		// --- 10. COHÉRENCE SIMD / SCALAIRE ---
		TEST_METHOD(SimdMatchesScalarVersion)
		{
			// Mêmes opérations sur la version SIMD (float) et la version générique (double)
			Mat simd = Mat::translate(1.5f, -2.5f) * Mat::rotation(0.7f) * Mat::scale(2.0f, 3.0f);
			MatD ref = MatD::translate(1.5, -2.5) * MatD::rotation(0.7) * MatD::scale(2.0, 3.0);

			for (int row = 0; row < 3; ++row)
				for (int col = 0; col < 3; ++col)
				{
					std::wstring msg = L"at(" + std::to_wstring(row) + L", " + std::to_wstring(col) + L")";
					Assert::AreEqual(static_cast<float>(ref.at(row, col)), simd.at(row, col), 1e-5f, msg.c_str());
				}

			// Même résultat sur un point
			V3 rs = simd.multiply(V3{ 3.0f, -4.0f, 1.0f });
			std::array<double, 3> rd = ref.multiply(std::array<double, 3>{ 3.0, -4.0, 1.0 });
			for (size_t i = 0; i < 3; ++i)
				Assert::AreEqual(static_cast<float>(rd[i]), rs[i], 1e-4f);
		}

		// --- 11. AFFICHAGE ---
		TEST_METHOD(Print)
		{
			std::ostringstream oss;
			std::streambuf* old = std::cout.rdbuf(oss.rdbuf());
			Mat::identity().print();
			std::cout.rdbuf(old);

			Assert::AreEqual(std::string("| 1 0 0 |\n| 0 1 0 |\n| 0 0 1 |\n"), oss.str());
		}
	};
}