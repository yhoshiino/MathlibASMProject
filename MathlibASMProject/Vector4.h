#pragma once
#include <xmmintrin.h>

struct Vector4
{
	Vector4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {};
	float x, z, y, w;
};

inline __m128 Load(const Vector4& v) {
	return _mm_setr_ps(v.x, v.y, v.z, v.w);
}



inline Vector4 Store(__m128 val) {
	float components[4];
	_mm_storeu_ps(components, val);

	return {
		components[0],
		components[1],
		components[2],
		components[3],
	};
}

inline Vector4 Add(Vector4& a, Vector4& b) {
	const __m128 va = Load(a);
	const __m128 vb = Load(b);

	const __m128 result = _mm_add_ps(va, vb);

	return Store(result);
}

inline Vector4 Scale(Vector4& a, float b) {
	const __m128 va = Load(a);
	const __m128 vb = _mm_set1_ps(b);

	return Store(_mm_mul_ps(va, vb));
}

inline Vector4 Lerp(const Vector4& a, const Vector4& b, float t) {
	const __m128 va = Load(a);
	const __m128 vb = Load(b);
	const __m128 ft = _mm_set1_ps(t);

	return Store(_mm_add_ps(va, _mm_mul_ps(_mm_sub_ps(vb, va), ft)));

}