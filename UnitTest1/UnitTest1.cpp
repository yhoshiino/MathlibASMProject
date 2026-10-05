#include "pch.h"
#include "CppUnitTest.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include <array>
#include "../MathLibrary/Vector3.h" // Vérifie le chemin vers Vector3.h

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace math;

namespace Vector3Tests
{
	TEST_CLASS(Vector3Tests)
	{
	public:

		// --- 1. CONSTRUCTEURS & ACCÈS PAR INDEX ---
		TEST_METHOD(ConstructorsAndAccess)
		{
			Vector3 defaultVec;
			Assert::AreEqual(0.0f, defaultVec.x);
			Assert::AreEqual(0.0f, defaultVec.y);
			Assert::AreEqual(0.0f, defaultVec.z);

			Vector3 paramVec(3.0f, 4.0f, 5.0f);
			Assert::AreEqual(3.0f, paramVec.x);
			Assert::AreEqual(4.0f, paramVec.y);
			Assert::AreEqual(5.0f, paramVec.z);

			// Operator[]
			Assert::AreEqual(3.0f, paramVec[0]);
			Assert::AreEqual(4.0f, paramVec[1]);
			Assert::AreEqual(5.0f, paramVec[2]);

			// Exceptions
			auto funcOutUpper = [&]() { paramVec[3]; };
			Assert::ExpectException<std::out_of_range>(funcOutUpper);

			auto funcOutLower = [&]() { paramVec[-1]; };
			Assert::ExpectException<std::out_of_range>(funcOutLower);
		}

		// --- 2. VECTEURS STATIQUES PRÉDÉFINIS ---
		TEST_METHOD(StaticProperties)
		{
			Assert::IsTrue(Vector3::up() == Vector3(0.0f, 1.0f, 0.0f));
			Assert::IsTrue(Vector3::down() == Vector3(0.0f, -1.0f, 0.0f));
			Assert::IsTrue(Vector3::left() == Vector3(-1.0f, 0.0f, 0.0f));
			Assert::IsTrue(Vector3::right() == Vector3(1.0f, 0.0f, 0.0f));
			Assert::IsTrue(Vector3::forward() == Vector3(0.0f, 0.0f, 1.0f));
			Assert::IsTrue(Vector3::back() == Vector3(0.0f, 0.0f, -1.0f));
			Assert::IsTrue(Vector3::one() == Vector3(1.0f, 1.0f, 1.0f));
			Assert::IsTrue(Vector3::zero() == Vector3(0.0f, 0.0f, 0.0f));

			Vector3 negInf = Vector3::negativeInfinity();
			Assert::IsTrue(std::isinf(negInf.x) && negInf.x < 0);
			Assert::IsTrue(std::isinf(negInf.y) && negInf.y < 0);
			Assert::IsTrue(std::isinf(negInf.z) && negInf.z < 0);

			Vector3 posInf = Vector3::positiveInfinity();
			Assert::IsTrue(std::isinf(posInf.x) && posInf.x > 0);
			Assert::IsTrue(std::isinf(posInf.y) && posInf.y > 0);
			Assert::IsTrue(std::isinf(posInf.z) && posInf.z > 0);
		}

		// --- 3. OPÉRATEURS ARITHMÉTIQUES ET D'AFFECTATION ---
		TEST_METHOD(ArithmeticOperators)
		{
			Vector3 a(2.0f, 3.0f, 4.0f);
			Vector3 b(4.0f, 1.0f, 2.0f);

			// Opérateurs binaires
			Assert::IsTrue((a + b) == Vector3(6.0f, 4.0f, 6.0f));
			Assert::IsTrue((a - b) == Vector3(-2.0f, 2.0f, 2.0f));
			Assert::IsTrue((a * 2.0f) == Vector3(4.0f, 6.0f, 8.0f));
			Assert::IsTrue((a / 2.0f) == Vector3(1.0f, 1.5f, 2.0f));

			// Comparaisons
			Assert::IsTrue(a == Vector3(2.0f, 3.0f, 4.0f));
			Assert::IsTrue(a != b);

			// Opérateurs d'affectation
			Vector3 v(1.0f, 2.0f, 3.0f);
			v += b;
			Assert::IsTrue(v == Vector3(5.0f, 3.0f, 5.0f));
			v -= a;
			Assert::IsTrue(v == Vector3(3.0f, 0.0f, 1.0f));
			v *= 2.0f;
			Assert::IsTrue(v == Vector3(6.0f, 0.0f, 2.0f));
			v /= 2.0f;
			Assert::IsTrue(v == Vector3(3.0f, 0.0f, 1.0f));
		}

		// --- 4. PRODUIT SCALAIRE, PRODUIT VECTORIEL & MAGNITUDES ---
		TEST_METHOD(MagnitudeDotAndCross)
		{
			Vector3 v(1.0f, 2.0f, 2.0f);

			Assert::AreEqual(9.0f, v.sqrMagnitude(), 0.0001f);
			Assert::AreEqual(3.0f, v.magnitude(), 0.0001f);

			// Produit scalaire (Dot) : 1*4 + 2*(-5) + 2*6 = 4 - 10 + 12 = 6
			Assert::AreEqual(6.0f, v.dot(Vector3(4.0f, -5.0f, 6.0f)), 0.0001f);

			// Produit vectoriel (Cross) : Right (1,0,0) x Up (0,1,0) = Forward (0,0,1)
			Vector3 crossResult = Vector3::right().cross(Vector3::up());
			Assert::IsTrue(crossResult == Vector3::forward());

			// Normalisation
			Vector3 vNorm(0.0f, 3.0f, 4.0f);
			Vector3 norm = vNorm.normalized();
			Assert::AreEqual(0.0f, norm.x, 0.0001f);
			Assert::AreEqual(0.6f, norm.y, 0.0001f);
			Assert::AreEqual(0.8f, norm.z, 0.0001f);
			Assert::AreEqual(1.0f, norm.magnitude(), 0.0001f);

			// Normalisation vecteur nul
			Vector3 zeroVec(0.0f, 0.0f, 0.0f);
			Assert::IsTrue(zeroVec.normalized() == Vector3::zero());
		}

		// --- 5. DISTANCE ET INTERPOLATION ---
		TEST_METHOD(DistanceAndInterpolation)
		{
			Vector3 a(0.0f, 0.0f, 0.0f);
			Vector3 b(0.0f, 3.0f, 4.0f);

			// Distance
			Assert::AreEqual(5.0f, Vector3::distance(a, b), 0.0001f);

			// Lerp (Clamped)
			Vector3 start(0.0f, 0.0f, 0.0f);
			Vector3 end(10.0f, 20.0f, 30.0f);

			Assert::IsTrue(Vector3::Lerp(start, end, 0.5f) == Vector3(5.0f, 10.0f, 15.0f));
			Assert::IsTrue(Vector3::Lerp(start, end, -0.5f) == start);
			Assert::IsTrue(Vector3::Lerp(start, end, 1.5f) == end);

			// LerpUnclamped
			Assert::IsTrue(Vector3::LerpUnclamped(start, end, 1.5f) == Vector3(15.0f, 30.0f, 45.0f));

			// Min / Max
			Vector3 v1(5.0f, 2.0f, 9.0f);
			Vector3 v2(3.0f, 8.0f, 1.0f);
			Assert::IsTrue(Vector3::Min(v1, v2) == Vector3(3.0f, 2.0f, 1.0f));
			Assert::IsTrue(Vector3::Max(v1, v2) == Vector3(5.0f, 8.0f, 9.0f));
		}

		// --- 6. DÉPLACEMENT ET TRANSFORMATIONS GÉOMÉTRIQUES ---
		TEST_METHOD(MovementAndGeometry)
		{
			// MoveTowards
			Vector3 current(0.0f, 0.0f, 0.0f);
			Vector3 target(10.0f, 0.0f, 0.0f);

			Vector3 step1 = Vector3::MoveTowards(current, target, 3.0f);
			Assert::IsTrue(step1 == Vector3(3.0f, 0.0f, 0.0f));

			Vector3 step2 = Vector3::MoveTowards(current, target, 15.0f);
			Assert::IsTrue(step2 == target);

			// Scale
			Vector3 v(3.0f, 4.0f, 5.0f);
			Assert::IsTrue(v.Scale(Vector3(2.0f, 0.5f, 3.0f)) == Vector3(6.0f, 2.0f, 15.0f));

			// Reflect
			Vector3 inRay(1.0f, -1.0f, 0.0f);
			Vector3 normal(0.0f, 1.0f, 0.0f);
			Assert::IsTrue(inRay.Reflect(normal) == Vector3(1.0f, 1.0f, 0.0f));

			// ClampMagnitude
			Vector3 longVec(10.0f, 0.0f, 0.0f);
			Vector3 clamped = longVec.ClampMagnitude(5.0f);
			Assert::AreEqual(5.0f, clamped.magnitude(), 0.0001f);
			Assert::IsTrue(clamped == Vector3(5.0f, 0.0f, 0.0f));

			Vector3 shortVec(2.0f, 0.0f, 0.0f);
			Assert::IsTrue(shortVec.ClampMagnitude(5.0f) == shortVec);
		}

		// --- 7. MUTATEURS, CONVERSIONS ET STRINGS ---
		TEST_METHOD(MutatorsAndConversions)
		{
			Vector3 v(1.0f, 2.0f, 3.0f);

			v.SetVector3(10.0f, 20.0f, 30.0f);
			Assert::AreEqual(10.0f, v.x);
			Assert::AreEqual(20.0f, v.y);
			Assert::AreEqual(30.0f, v.z);

			Assert::AreEqual(std::string("(10.000000, 20.000000, 30.000000)"), v.toString());

			std::array<float, 3> arr = v.toArray();
			Assert::AreEqual(10.0f, arr[0]);
			Assert::AreEqual(20.0f, arr[1]);
			Assert::AreEqual(30.0f, arr[2]);
		}
	};
}