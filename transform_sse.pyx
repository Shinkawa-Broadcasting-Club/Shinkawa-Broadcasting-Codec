# cython: boundscheck=False, wraparound=False, language_level=3
cimport cython
from cython.parallel import prange
cdef extern from "nmmintrin.h" nogil:
	ctypedef struct __m128: pass
	ctypedef struct __m128i: pass
	__m128i _mm_loadl_epi64 (const __m128i* mem_addr)
	__m128i _mm_cvtepu16_epi32 (__m128i a)
	__m128 _mm_loadu_ps (const float* mem_addr)
	__m128 _mm_cvtepi32_ps (__m128i a)
	void _mm_storeu_ps (float* mem_addr, __m128 a)

# 符号無16bit整数->符号付32bit浮動小数点数
cdef inline float* ui16_fp32(unsigned short* arr_in, float* arr_out) nogil:
	cdef unsigned char n
	for n in prange(8): _mm_storeu_ps(arr_out + (n << 2), _mm_cvtepi32_ps(_mm_cvtepu16_epi32(_mm_loadl_epi64(<const __m128i*>(arr_in + (n << 2))))))
	return arr_out

# DCT(離散余弦変換)
cdef inline float* dct_fwd(float* arr_in, float* arr_out) nogil:
	cdef __m128 n0, n1, n2, n3, n4, n5, n6, n7, v0, v1, v2, v3, v4, v5, v6, v7
	return arr_out

cdef inline void dct_3d_fwd(unsigned short[32][32][32] arr, float[32][32][32] out) nogil:
	cdef:
		float[32][32][32] tmp0, tmp1
		unsigned char l, m, n
	# 符号無16bit整数->符号付32bit浮動小数点数
	for l in prange(32):
		for m in prange(32):
			ui16_fp32(&arr[l][m][0], &tmp0[l][m][0])
	# 横方向にDCT
	for l in prange(32):
		for m in prange(32):
			dct_fwd(&tmp0[l][m][0], &tmp1[l][m][0])
	# (時、縦、横)->(横、時、縦)
	for l in prange(32):
		for m in prange(32):
			for n in prange(32):
				tmp0[l][m][n] = tmp1[n][l][m]
	# 縦方向にDCT
	for l in prange(32):
		for m in prange(32):
			dct_fwd(&tmp0[l][m][0], &tmp1[l][m][0])
	# (横、時、縦)->(縦、横、時)
	for l in prange(32):
		for m in prange(32):
			for n in prange(32):
				tmp0[l][m][n] = tmp1[n][l][m]