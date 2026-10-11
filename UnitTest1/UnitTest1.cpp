// ============================================================================
// Vector3SIMDTests.cpp
//
// FICHIER TESTÉ : MathLibrary/Vector3SIMD.h   (classe math::simd::Vector3)
//
// Dans l'Explorateur de tests de Visual Studio, tous les tests de ce fichier sont
// regroupés sous la classe "Vector3SIMD_Tests". Le début du nom d'un test indique
// la fonction testée :
//
//   Constructors_*   ->  Vector3(), Vector3(x, y, z), Vector3(__m128)
//   Layout_*         ->  taille 16 octets, alignement 16, union reg / (x, y, z, w)
//   Dot_*            ->  Vector3::dot
//   Magnitude_*      ->  Vector3::magnitude
//   Normalized_*     ->  Vector3::normalized
//   Detail_*         ->  detail::dot_xyz et detail::mask_xyz (fonctions internes)
//   Constant_*       ->  kNormalizeEpsSq
//
// Les tests de Mat4x4f sont dans Mat4x4SIMDTests.cpp.
// Les traitements par lots (BatchOps) ne sont PAS testés ici.
// ============================================================================
#include "pch.h"
#include "CppUnitTest.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <emmintrin.h>
#include "../MathLibrary/Vector3SIMD.h" // Vérifie le chemin vers Vector3SIMD.h

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace math::simd;

namespace
{
	// Tolérance mixte (cahier des charges : absolue + relative) :
	//   |attendu - obtenu| <= absTol + relTol * |attendu|
	// - absTol = 1e-6 : valeurs proches de 0 (l'epsilon du float vaut 1,2e-7 : on laisse quelques opérations)
	// - relTol = 1e-5 : grandes valeurs (environ 80 fois l'epsilon, assez pour sqrt + division)
	constexpr float ABS_TOL = 1e-6f;
	constexpr float REL_TOL = 1e-5f;

	float Tol(float expected)
	{
		return ABS_TOL + REL_TOL * std::abs(expected);
	}

	void AssertNear(float expected, float actual, const wchar_t* msg = nullptr)
	{
		Assert::AreEqual(expected, actual, Tol(expected), msg);
	}

	// Vecteur dont on choisit aussi la lane w (le constructeur (x, y, z) la met toujours à 0)
	Vector3 WithW(float x, float y, float z, float w)
	{
		return Vector3(_mm_setr_ps(x, y, z, w));
	}

	// Longueur calculée en double précision : référence indépendante du code testé
	double LengthD(float x, float y, float z)
	{
		return std::sqrt(static_cast<double>(x) * x + static_cast<double>(y) * y + static_cast<double>(z) * z);
	}
}

namespace Vector3SIMDTests
{
	TEST_CLASS(Vector3SIMD_Tests)
	{
	public:

		// ====================================================================
		// Constructors_*  -> Vector3(), Vector3(x, y, z), Vector3(__m128)
		// ====================================================================

		TEST_METHOD(Constructors_Default_AllLanesZero)
		{
			Vector3 v;
			Assert::AreEqual(0.0f, v.x);
			Assert::AreEqual(0.0f, v.y);
			Assert::AreEqual(0.0f, v.z);
			Assert::AreEqual(0.0f, v.w);
		}

		TEST_METHOD(Constructors_XYZ_SetsComponentsAndPaddingToZero)
		{
			Vector3 v(1.5f, -2.5f, 3.5f);
			Assert::AreEqual(1.5f, v.x);
			Assert::AreEqual(-2.5f, v.y);
			Assert::AreEqual(3.5f, v.z);
			Assert::AreEqual(0.0f, v.w); // w est le padding : toujours 0 avec ce constructeur
		}

		TEST_METHOD(Constructors_FromRegister_KeepsAllFourLanes)
		{
			// Ce constructeur copie le registre tel quel : w n'est PAS forcé à 0
			Vector3 v(_mm_setr_ps(1.0f, 2.0f, 3.0f, 5.0f));
			Assert::AreEqual(1.0f, v.x);
			Assert::AreEqual(2.0f, v.y);
			Assert::AreEqual(3.0f, v.z);
			Assert::AreEqual(5.0f, v.w);
		}

		// ====================================================================
		// Layout_*  -> taille, alignement, union reg / x y z w
		// ====================================================================

		TEST_METHOD(Layout_SizeAndAlignment_16Bytes)
		{
			Assert::AreEqual(static_cast<size_t>(16), sizeof(Vector3));
			Assert::AreEqual(static_cast<size_t>(16), alignof(Vector3));

			Vector3 v;
			Assert::AreEqual(static_cast<std::uintptr_t>(0), reinterpret_cast<std::uintptr_t>(&v) % 16);
		}

		TEST_METHOD(Layout_ArrayAndStdVector_AreAligned)
		{
			// BatchOps suppose que ses tableaux de Vector3 sont alignés sur 16 octets
			Vector3 arr[4];
			for (int i = 0; i < 4; ++i)
				Assert::AreEqual(static_cast<std::uintptr_t>(0), reinterpret_cast<std::uintptr_t>(&arr[i]) % 16);

			std::vector<Vector3> vec(5);
			Assert::AreEqual(static_cast<std::uintptr_t>(0), reinterpret_cast<std::uintptr_t>(vec.data()) % 16);
			Assert::AreEqual(static_cast<std::uintptr_t>(0), reinterpret_cast<std::uintptr_t>(&vec[3]) % 16);
		}

		TEST_METHOD(Layout_RegisterAndComponents_ShareTheSameMemory)
		{
			Vector3 v(1.0f, 2.0f, 3.0f);
			alignas(16) float out[4];

			// reg contient (x, y, z, w) dans cet ordre
			_mm_store_ps(out, v.reg);
			Assert::AreEqual(1.0f, out[0]);
			Assert::AreEqual(2.0f, out[1]);
			Assert::AreEqual(3.0f, out[2]);
			Assert::AreEqual(0.0f, out[3]);

			// Modifier un composant modifie le registre
			v.y = 9.0f;
			_mm_store_ps(out, v.reg);
			Assert::AreEqual(9.0f, out[1]);
		}

		// ====================================================================
		// Dot_*  -> Vector3::dot
		// ====================================================================

		TEST_METHOD(Dot_KnownValues)
		{
			// Valeurs calculées à la main
			Assert::AreEqual(32.0f, Vector3(1, 2, 3).dot(Vector3(4, 5, 6)), 1e-6f); // 4 + 10 + 18
			Assert::AreEqual(1.0f, Vector3(1, 0, 0).dot(Vector3(1, 0, 0)), 1e-6f);
			Assert::AreEqual(0.0f, Vector3(1, 0, 0).dot(Vector3(0, 1, 0)), 1e-6f);  // perpendiculaires
			Assert::AreEqual(-1.0f, Vector3(1, 0, 0).dot(Vector3(-1, 0, 0)), 1e-6f); // opposés
			Assert::AreEqual(0.0f, Vector3(2, 2, 2).dot(Vector3(1, -1, 0)), 1e-6f);
			Assert::AreEqual(0.0f, Vector3(0, 0, 0).dot(Vector3(7, 8, 9)), 1e-6f);   // vecteur nul
		}

		TEST_METHOD(Dot_IsCommutative)
		{
			Vector3 a(1.5f, -2.0f, 3.25f);
			Vector3 b(-4.0f, 0.5f, 2.0f);
			Assert::AreEqual(a.dot(b), b.dot(a), 1e-6f);
		}

		TEST_METHOD(Dot_ScalingAVectorScalesTheResult)
		{
			Vector3 a(1, 2, 3);
			Vector3 b(4, 5, 6);
			Vector3 a3(3, 6, 9); // a * 3
			AssertNear(3.0f * a.dot(b), a3.dot(b));
		}

		TEST_METHOD(Dot_IgnoresLaneW)
		{
			// Le padding w ne doit jamais influencer le résultat
			Vector3 a = WithW(1, 2, 3, 999.0f);
			Vector3 b = WithW(4, 5, 6, -777.0f);
			Assert::AreEqual(32.0f, a.dot(b), 1e-6f);
		}

		TEST_METHOD(Dot_MatchesScalarFormula)
		{
			// Même ordre d'addition que la formule scalaire : (x*x' + y*y') + z*z'
			const std::array<std::array<float, 3>, 5> as = { {
				{ 1.0f, 2.0f, 3.0f }, { -5.0f, 0.5f, 12.0f }, { 100.0f, -200.0f, 300.0f },
				{ 1e-3f, 2e-3f, -3e-3f }, { 0.1f, 0.2f, 0.3f } } };
			const std::array<std::array<float, 3>, 5> bs = { {
				{ 4.0f, 5.0f, 6.0f }, { 2.0f, -7.0f, 0.25f }, { -3.0f, 1.0f, 0.5f },
				{ 5e-3f, 1e-3f, 1e-3f }, { 0.3f, 0.2f, 0.1f } } };
			for (size_t i = 0; i < as.size(); ++i)
			{
				const float expected = as[i][0] * bs[i][0] + as[i][1] * bs[i][1] + as[i][2] * bs[i][2];
				const float actual = Vector3(as[i][0], as[i][1], as[i][2]).dot(Vector3(bs[i][0], bs[i][1], bs[i][2]));
				AssertNear(expected, actual);
			}
		}

		// ====================================================================
		// Magnitude_*  -> Vector3::magnitude
		// ====================================================================

		TEST_METHOD(Magnitude_KnownValues)
		{
			Assert::AreEqual(5.0f, Vector3(3, 4, 0).magnitude(), 1e-6f);
			Assert::AreEqual(3.0f, Vector3(1, 2, 2).magnitude(), 1e-6f);
			Assert::AreEqual(7.0f, Vector3(0, 0, -7).magnitude(), 1e-6f);
			Assert::AreEqual(0.0f, Vector3(0, 0, 0).magnitude(), 1e-6f);
			Assert::AreEqual(std::sqrt(3.0f), Vector3(1, 1, 1).magnitude(), 1e-6f);
		}

		TEST_METHOD(Magnitude_IgnoresLaneW)
		{
			Assert::AreEqual(5.0f, WithW(3, 4, 0, 999.0f).magnitude(), 1e-6f);
		}

		TEST_METHOD(Magnitude_SquaredEqualsDotWithItself)
		{
			Vector3 v(1.5f, -2.5f, 4.0f);
			const float m = v.magnitude();
			AssertNear(v.dot(v), m * m);
		}

		TEST_METHOD(Magnitude_MatchesDoublePrecisionReference)
		{
			const std::array<std::array<float, 3>, 4> vs = { {
				{ -5.0f, 0.5f, 12.0f }, { 100.0f, -200.0f, 300.0f }, { 1e-3f, 2e-3f, -3e-3f }, { 0.1f, 0.2f, 0.3f } } };
			for (const auto& v : vs)
				AssertNear(static_cast<float>(LengthD(v[0], v[1], v[2])), Vector3(v[0], v[1], v[2]).magnitude());
		}

		// ====================================================================
		// Normalized_*  -> Vector3::normalized
		// ====================================================================

		TEST_METHOD(Normalized_KnownValues)
		{
			Vector3 a = Vector3(3, 4, 0).normalized();
			AssertNear(0.6f, a.x);
			AssertNear(0.8f, a.y);
			Assert::AreEqual(0.0f, a.z, 1e-6f);

			Vector3 b = Vector3(0, 0, -2).normalized();
			Assert::AreEqual(0.0f, b.x, 1e-6f);
			Assert::AreEqual(0.0f, b.y, 1e-6f);
			AssertNear(-1.0f, b.z);

			const float inv = 1.0f / std::sqrt(3.0f);
			Vector3 c = Vector3(1, 1, 1).normalized();
			AssertNear(inv, c.x);
			AssertNear(inv, c.y);
			AssertNear(inv, c.z);
		}

		TEST_METHOD(Normalized_HasUnitLengthAndKeepsDirection)
		{
			// Référence indépendante : composante / longueur calculée en double
			const std::array<std::array<float, 3>, 6> vs = { {
				{ 3.0f, 4.0f, 0.0f }, { 1.0f, 2.0f, 2.0f }, { -5.0f, 0.5f, 12.0f },
				{ 100.0f, -200.0f, 300.0f }, { 1e-3f, 2e-3f, -3e-3f }, { 0.0f, 0.0f, 9.0f } } };
			for (const auto& v : vs)
			{
				const double len = LengthD(v[0], v[1], v[2]);
				Vector3 n = Vector3(v[0], v[1], v[2]).normalized();
				AssertNear(static_cast<float>(v[0] / len), n.x);
				AssertNear(static_cast<float>(v[1] / len), n.y);
				AssertNear(static_cast<float>(v[2] / len), n.z);
				AssertNear(1.0f, n.magnitude());
			}
		}

		TEST_METHOD(Normalized_ResultLaneWIsZero)
		{
			// Même si w contient n'importe quoi en entrée, le résultat a w = 0
			Vector3 n = WithW(3, 4, 0, 999.0f).normalized();
			Assert::AreEqual(0.0f, n.w);
			AssertNear(0.6f, n.x);
			AssertNear(0.8f, n.y);
		}

		TEST_METHOD(Normalized_ZeroVector_ReturnsZeroNotNaN)
		{
			// Convention documentée : vecteur nul -> (0, 0, 0), jamais de NaN
			Vector3 n = Vector3(0, 0, 0).normalized();
			Assert::AreEqual(0.0f, n.x);
			Assert::AreEqual(0.0f, n.y);
			Assert::AreEqual(0.0f, n.z);
			Assert::AreEqual(0.0f, n.w);
			Assert::IsFalse(std::isnan(n.x) || std::isnan(n.y) || std::isnan(n.z));
		}

		TEST_METHOD(Normalized_BelowThreshold_ReturnsZero)
		{
			// |v|^2 = 1e-14 <= kNormalizeEpsSq (1e-12) : considéré comme nul
			Vector3 tiny = Vector3(1e-7f, 0.0f, 0.0f).normalized();
			Assert::AreEqual(0.0f, tiny.x);
			Assert::AreEqual(0.0f, tiny.y);
			Assert::AreEqual(0.0f, tiny.z);

			Vector3 tiny3 = Vector3(1e-7f, -2e-7f, 1e-7f).normalized();
			Assert::AreEqual(0.0f, tiny3.x);
			Assert::AreEqual(0.0f, tiny3.y);
			Assert::AreEqual(0.0f, tiny3.z);
		}

		TEST_METHOD(Normalized_SmallButAboveThreshold_IsNormalized)
		{
			// |v|^2 = 1e-6 > 1e-12 : petit mais valide
			Vector3 n = Vector3(1e-3f, 0.0f, 0.0f).normalized();
			AssertNear(1.0f, n.x);
			Assert::AreEqual(0.0f, n.y, 1e-6f);
			Assert::AreEqual(0.0f, n.z, 1e-6f);
		}

		TEST_METHOD(Normalized_NaNInput_ReturnsZero)
		{
			// La comparaison "len2 > seuil" est fausse pour NaN : le résultat est forcé à 0
			const float nan = std::numeric_limits<float>::quiet_NaN();
			Vector3 n = Vector3(nan, 1.0f, 1.0f).normalized();
			Assert::AreEqual(0.0f, n.x);
			Assert::AreEqual(0.0f, n.y);
			Assert::AreEqual(0.0f, n.z);
		}

		// ====================================================================
		// Detail_*  -> detail::dot_xyz, detail::mask_xyz
		// ====================================================================

		TEST_METHOD(Detail_DotXYZ_BroadcastsResultInAllLanes)
		{
			alignas(16) float out[4];
			_mm_store_ps(out, detail::dot_xyz(_mm_setr_ps(1, 2, 3, 999), _mm_setr_ps(4, 5, 6, 999)));
			for (int i = 0; i < 4; ++i)
				Assert::AreEqual(32.0f, out[i], 1e-6f); // 4 + 10 + 18, w ignoré, même valeur dans les 4 lanes
		}

		TEST_METHOD(Detail_MaskXYZ_KeepsXYZAndClearsW)
		{
			alignas(16) std::uint32_t bits[4];
			_mm_store_si128(reinterpret_cast<__m128i*>(bits), _mm_castps_si128(detail::mask_xyz()));
			Assert::AreEqual(0xFFFFFFFFu, bits[0]);
			Assert::AreEqual(0xFFFFFFFFu, bits[1]);
			Assert::AreEqual(0xFFFFFFFFu, bits[2]);
			Assert::AreEqual(0u, bits[3]);
		}

		// ====================================================================
		// Constant_*  -> kNormalizeEpsSq
		// ====================================================================

		TEST_METHOD(Constant_NormalizeEpsSq_IsOneEMinus12)
		{
			// Seuil documenté : |v| < 1e-6, soit |v|^2 <= 1e-12
			Assert::AreEqual(1e-12f, kNormalizeEpsSq);
		}
	};
}