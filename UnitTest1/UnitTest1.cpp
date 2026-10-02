#include "pch.h"
#include "CppUnitTest.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include "../MathLibrary/Vector2.h" // Assure-toi que le chemin vers Vector2.h est correct dans les propriétés de ton projet[cite: 1]

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace math;

namespace Vector2Tests
{
	TEST_CLASS(Vector2Tests)
	{
	public:

		// --- 1. CONSTRUCTEURS & ACCÈS PAR INDEX ---
		TEST_METHOD(ConstructorsAndAccess)
		{
			Vector2 defaultVec;
			Assert::AreEqual(0.0f, defaultVec.x);
			Assert::AreEqual(0.0f, defaultVec.y);

			Vector2 paramVec(3.0f, 4.0f);
			Assert::AreEqual(3.0f, paramVec.x);
			Assert::AreEqual(4.0f, paramVec.y);

			// Operator[]
			Assert::AreEqual(3.0f, paramVec[0]);
			Assert::AreEqual(4.0f, paramVec[1]);

			// Exceptions
			auto funcOutUpper = [&]() { paramVec[2]; };
			Assert::ExpectException<std::out_of_range>(funcOutUpper);

			auto funcOutLower = [&]() { paramVec[-1]; };
			Assert::ExpectException<std::out_of_range>(funcOutLower);
		}

		// --- 2. VECTEURS STATIQUES PRÉDÉFINIS ---
		TEST_METHOD(StaticProperties)
		{
			Assert::IsTrue(Vector2::up() == Vector2(0.0f, 1.0f));
			Assert::IsTrue(Vector2::down() == Vector2(0.0f, -1.0f));
			Assert::IsTrue(Vector2::left() == Vector2(-1.0f, 0.0f));
			Assert::IsTrue(Vector2::right() == Vector2(1.0f, 0.0f));
			Assert::IsTrue(Vector2::one() == Vector2(1.0f, 1.0f));

			Vector2 negInf = Vector2::negativeInfinity();
			Assert::IsTrue(std::isinf(negInf.x) && negInf.x < 0);
			Assert::IsTrue(std::isinf(negInf.y) && negInf.y < 0);

			Vector2 posInf = Vector2::positiveInfinity();
			Assert::IsTrue(std::isinf(posInf.x) && posInf.x > 0);
			Assert::IsTrue(std::isinf(posInf.y) && posInf.y > 0);
		}

		// --- 3. OPÉRATEURS ARITHMÉTIQUES ET D'AFFECTATION ---
		TEST_METHOD(ArithmeticOperators)
		{
			Vector2 a(2.0f, 3.0f);
			Vector2 b(4.0f, 1.0f);

			// Opérateurs binaires
			Assert::IsTrue((a + b) == Vector2(6.0f, 4.0f));
			Assert::IsTrue((a - b) == Vector2(-2.0f, 2.0f));
			Assert::IsTrue((a * 2.0f) == Vector2(4.0f, 6.0f));
			Assert::IsTrue((a / 2.0f) == Vector2(1.0f, 1.5f));

			// Opérateurs d'affectation
			Vector2 v(1.0f, 2.0f);
			v += b;
			Assert::IsTrue(v == Vector2(5.0f, 3.0f));
			v -= a;
			Assert::IsTrue(v == Vector2(3.0f, 0.0f));
			v *= 3.0f;
			Assert::IsTrue(v == Vector2(9.0f, 0.0f));
			v /= 3.0f;
			Assert::IsTrue(v == Vector2(3.0f, 0.0f));
		}

		// --- 4. PRODUIT SCALAIRE, MAGNITUDE ET NORMALISATION ---
		TEST_METHOD(MagnitudeAndDot)
		{
			Vector2 v(3.0f, 4.0f);

			Assert::AreEqual(25.0f, v.sqrMagnitude(), 0.0001f);
			Assert::AreEqual(5.0f, v.magnitude(), 0.0001f);
			Assert::AreEqual(2.0f, v.dot(Vector2(2.0f, -1.0f)), 0.0001f); // 3*2 + 4*(-1) = 2[cite: 1]

			// Normalisation
			Vector2 norm = v.normalized();
			Assert::AreEqual(0.6f, norm.x, 0.0001f);
			Assert::AreEqual(0.8f, norm.y, 0.0001f);
			Assert::AreEqual(1.0f, norm.magnitude(), 0.0001f);

			// Normalisation vecteur nul
			Vector2 zeroVec(0.0f, 0.0f);
			Assert::IsTrue(zeroVec.normalized() == Vector2(0.0f, 0.0f));
		}

		// --- 5. ANGLES ET DISTANCE ---
		TEST_METHOD(DistanceAndAngle)
		{
			Vector2 a(0.0f, 0.0f);
			Vector2 b(3.0f, 4.0f);

			// Distance
			Assert::AreEqual(5.0f, Vector2::distance(a, b), 0.0001f);

			// Angle (0 à 180)
			Vector2 vRight = Vector2::right();
			Vector2 vUp = Vector2::up();
			Assert::AreEqual(90.0f, Vector2::angle(vRight, vUp), 0.0001f);
			Assert::AreEqual(180.0f, Vector2::angle(vRight, Vector2::left()), 0.0001f);
			Assert::AreEqual(0.0f, Vector2::angle(a, b), 0.0001f); // Cas zéro[cite: 1]

			// SignedAngle (-180 à 180)
			Assert::AreEqual(90.0f, Vector2::SignedAngle(vRight, vUp), 0.0001f);
			Assert::AreEqual(-90.0f, Vector2::SignedAngle(vUp, vRight), 0.0001f);
		}

		// --- 6. INTERPOLATION ET LIMITES ---
		TEST_METHOD(InterpolationAndLimits)
		{
			Vector2 start(0.0f, 0.0f);
			Vector2 end(10.0f, 20.0f);

			// Lerp (Clamped)
			Assert::IsTrue(Vector2::Lerp(start, end, 0.5f) == Vector2(5.0f, 10.0f));
			Assert::IsTrue(Vector2::Lerp(start, end, -0.5f) == start);
			Assert::IsTrue(Vector2::Lerp(start, end, 1.5f) == end);

			// LerpUnclamped
			Assert::IsTrue(Vector2::LerpUnclamped(start, end, 1.5f) == Vector2(15.0f, 30.0f));

			// Min / Max
			Vector2 v1(5.0f, 2.0f);
			Vector2 v2(3.0f, 8.0f);
			Assert::IsTrue(Vector2::Min(v1, v2) == Vector2(3.0f, 2.0f));
			Assert::IsTrue(Vector2::Max(v1, v2) == Vector2(5.0f, 8.0f));
		}

		// --- 7. DÉPLACEMENT ET PHYSIQUE ---
		TEST_METHOD(MovementAndPhysics)
		{
			// MoveTowards
			Vector2 current(0.0f, 0.0f);
			Vector2 target(10.0f, 0.0f);

			Vector2 step1 = Vector2::MoveTowards(current, target, 3.0f);
			Assert::IsTrue(step1 == Vector2(3.0f, 0.0f));

			Vector2 step2 = Vector2::MoveTowards(current, target, 15.0f);
			Assert::IsTrue(step2 == target);

			// SmoothDamp
			Vector2 vel(0.0f, 0.0f);
			Vector2 nextPos = Vector2::SmoothDamp(current, target, vel, 0.1f, 0.02f);
			Assert::IsTrue(nextPos.x > 0.0f && nextPos.x < target.x);
			Assert::IsTrue(vel.x > 0.0f);
		}

		// --- 8. GÉOMÉTRIE ET TRANSFORMATIONS ---
		TEST_METHOD(Geometry)
		{
			Vector2 v(3.0f, 4.0f);

			// Perpendicular (-y, x)
			Assert::IsTrue(v.Perpendicular() == Vector2(-4.0f, 3.0f));

			// Scale
			Assert::IsTrue(v.Scale(Vector2(2.0f, 0.5f)) == Vector2(6.0f, 2.0f));

			// Reflect
			Vector2 inRay(1.0f, -1.0f);
			Vector2 normal(0.0f, 1.0f);
			Assert::IsTrue(inRay.Reflect(normal) == Vector2(1.0f, 1.0f));

			// Clampmagnitude
			Vector2 longVec(10.0f, 0.0f);
			Vector2 clamped = v.ClampMagnitude(longVec, 5.0f);
			Assert::AreEqual(5.0f, clamped.magnitude(), 0.0001f);
			Assert::IsTrue(clamped == Vector2(5.0f, 0.0f));

			Vector2 shortVec(2.0f, 0.0f);
			Assert::IsTrue(v.ClampMagnitude(shortVec, 5.0f) == shortVec);
		}

		// --- 9. MUTATEURS ET CHAINES DE CARACTÈRES ---
		TEST_METHOD(MutatorsAndString)
		{
			Vector2 v(1.0f, 2.0f);

			v.SetVector2(10.0f, 20.0f);
			Assert::AreEqual(10.0f, v.x);
			Assert::AreEqual(20.0f, v.y);

			Assert::AreEqual(std::string("(10.000000, 20.000000)"), v.toString());
		}
	};
}