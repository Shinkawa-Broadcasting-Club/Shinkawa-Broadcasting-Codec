# cython: boundscheck=False, wraparound=False, language_level=3
cimport cython
from cython.parallel import prange
cdef extern from "nmmintrin.h" nogil:
	ctypedef struct __m128: pass
	ctypedef struct __m128i: pass
	__m128i _mm_loadl_epi64 (const __m128i* mem_addr)
	__m128i _mm_cvtepu16_epi32 (__m128i a)
	__m128i _mm_set_epi32 (int e3, int e2, int e1, int e0)
	__m128 _mm_load1_ps (const float* mem_addr)
	__m128 _mm_castsi128_ps (__m128i a)
	__m128 _mm_loadu_ps (const float* mem_addr)
	__m128 _mm_cvtepi32_ps (__m128i a)
	__m128 _mm_add_ps (__m128 a, __m128 b)
	__m128 _mm_sub_ps (__m128 a, __m128 b)
	__m128 _mm_mul_ps (__m128 a, __m128 b)
	void _mm_storeu_ps (float* mem_addr, __m128 a)

# 定数代入(32bit浮動小数点数、最高精度)
cdef inline __m128 _mm_setconst_ps(int c) nogil: return _mm_load1_ps(<const float*> c)

# 時間軸DCT(離散余弦変換)
cdef inline void dct_time_fwd(__m128[8] arr_in, __m128[8] arr_out) nogil:
	cdef:
		__m128[8] tmp0, tmp1
		unsigned char n, m
	for m in range(3):
		for n in prange(8):
			if m == 0:    tmp0[n] = _mm_add_ps(arr_in[n], arr_in[n + 4]) if n & 7 < 4 else _mm_sub_ps(arr_in[n - 4], arr_in[n]) # stage 1
			if m == 1:    tmp1[n] = _mm_add_ps(  tmp0[n],   tmp0[n + 2]) if n & 3 < 2 else _mm_sub_ps(  tmp0[n - 2],   tmp0[n]) # stage 2
			if m == 2: arr_out[n] = _mm_add_ps(  tmp1[n],   tmp1[n + 1]) if n & 1 < 1 else _mm_sub_ps(  tmp1[n - 1],   tmp1[n]) # stage 3

# 空間DCT
cdef inline void dct_fwd(__m128[8] arr_in, __m128[8] arr_out) nogil:
	cdef:
		__m128[8] tmp0, tmp1
		__m128[4] a
		__m128 tmp
		unsigned char n
	
	# stage 1
	for n in prange(8):
		if n < 4: tmp0[n] = _mm_add_ps(arr_in[n], arr_in[7 - n])
		else:     tmp0[n] = _mm_sub_ps(arr_in[n], arr_in[7 - n])
	# stage 2
	for n in prange(8):
		if n < 2:     tmp1[n] = _mm_add_ps(tmp0[n], tmp0[3 - n])
		if 1 < n < 4: tmp1[n] = _mm_sub_ps(tmp0[n], tmp0[3 - n])
		if 3 < n < 7: tmp1[n] = _mm_add_ps(tmp0[n], tmp0[n + 1])
	# stage 3
	for n in prange(9):
		if n == 0: tmp0[0] = _mm_add_ps(tmp1[0], tmp1[1])
		if n == 1: tmp0[1] = _mm_sub_ps(tmp1[0], tmp1[1])
		if n == 2: tmp0[2] = _mm_mul_ps(_mm_add_ps(tmp1[2], tmp1[3]), _mm_setconst_ps(1060439283))
		if n == 4: tmp0[4] = _mm_mul_ps(tmp1[4], _mm_setconst_ps(1057655764))
		if n == 5: tmp0[5] = _mm_mul_ps(tmp0[5], _mm_setconst_ps(1060439283))
		if n == 6: tmp0[5] = _mm_mul_ps(tmp0[6], _mm_setconst_ps(1067924853))
		if n == 8: tmp = _mm_mul_ps(_mm_sub_ps(tmp1[6], tmp1[4]), _mm_setconst_ps(1053028117))

cdef inline void dct_3d_fwd(unsigned short[8][8][8] arr, float[8][8][8] out) nogil:
	cdef:
		__m128[8][8][2] tmp
		unsigned char l, m, n