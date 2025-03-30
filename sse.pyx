import numpy as np
import cython
cimport cython
cimport numpy as cnp
cdef extern from *:
	"""
	#include <xmmintrin.h>
	#include <emmintrin.h>
	"""
	ctypedef struct __m128i: pass # 32ビット整数 x 4
	ctypedef struct __m128:	pass # 32ビット浮動小数点数 x 4
	__m128i _mm_set_epi16(short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
	__m128i _mm_set_epi32(int e3, int e2, int e1, int e0)
	__m128i _mm_add_epi32(__m128i a, __m128i b)
	__m128i _mm_sub_epi32(__m128i a, __m128i b)
	__m128 _mm_cvtepi32_ps(__m128i a)
	void _mm_store_si128(__m128i* p, __m128i a)
	__m128 _mm_store_ps(float *p, __m128 a)

# 4つの符号無16ビット整数をそれぞれ足し、結果を符号有32ビット整数に格納する
@cython.boundscheck(False)
@cython.wraparound(False)
cdef add_short_to_int(cnp.ndarray[cnp.uint16_t, ndim = 1] a, cnp.ndarray[cnp.uint16_t, ndim = 1] b):
	cdef unsigned short s = 0
	cdef __m128i longa = _mm_set_epi16(a[3], s, a[2], s, a[1], s, a[0], s)
	cdef __m128i longb = _mm_set_epi16(b[3], s, b[2], s, b[1], s, b[0], s)
	cdef __m128i longc = _mm_add_epi32(longa, longb)
	cdef cnp.ndarray[cnp.int32_t, ndim=1] result = np.empty(4)
	cdef unsigned int* pdata = <unsigned int*> result.data
	_mm_store_si128(<__m128i *> pdata, longc)
	return result

# 4つの符号無16ビット整数をそれぞれ引き、結果を符号有32ビット整数に格納する
@cython.boundscheck(False)
@cython.wraparound(False)
cdef sub_short_to_int(cnp.ndarray[cnp.uint16_t, ndim = 1] a, cnp.ndarray[cnp.uint16_t, ndim = 1] b):
	cdef unsigned short s = 0
	cdef __m128i longa = _mm_set_epi16(a[3], s, a[2], s, a[1], s, a[0], s)
	cdef __m128i longb = _mm_set_epi16(b[3], s, b[2], s, b[1], s, b[0], s)
	cdef __m128i longc = _mm_sub_epi32(longa, longb)
	cdef cnp.ndarray[cnp.int32_t, ndim=1] result = np.empty(4)
	cdef unsigned int* pdata = <unsigned int*> result.data
	_mm_store_si128(<__m128i *> pdata, longc)
	return result

# 4つの符号有32ビット整数をそれぞれ足し、結果を符号有32ビット整数に格納する
@cython.boundscheck(False)
@cython.wraparound(False)
cdef add_int(cnp.ndarray[cnp.int32_t, ndim = 1] a, cnp.ndarray[cnp.int32_t, ndim = 1] b):
	cdef __m128i longa = _mm_set_epi32(a[3], a[2], a[1], a[0])
	cdef __m128i longb = _mm_set_epi32(b[3], b[2], b[1], b[0])
	cdef __m128i longc = _mm_add_epi32(longa, longb)
	cdef cnp.ndarray[cnp.int32_t, ndim=1] result = np.empty(4)
	cdef unsigned int* pdata = <unsigned int*> result.data
	_mm_store_si128(<__m128i *> pdata, longc)
	return result

# 4つの符号有32ビット整数をそれぞれ引き、結果を符号有32ビット整数に格納する
@cython.boundscheck(False)
@cython.wraparound(False)
cdef sub_int(cnp.ndarray[cnp.int32_t, ndim = 1] a, cnp.ndarray[cnp.int32_t, ndim = 1] b):
	cdef __m128i longa = _mm_set_epi32(a[3], a[2], a[1], a[0])
	cdef __m128i longb = _mm_set_epi32(b[3], b[2], b[1], b[0])
	cdef __m128i longc = _mm_sub_epi32(longa, longb)
	cdef cnp.ndarray[cnp.int32_t, ndim=1] result = np.empty(4)
	cdef unsigned int* pdata = <unsigned int*> result.data
	_mm_store_si128(<__m128i *> pdata, longc)
	return result

# 4つの符号有32ビット整数をそれぞれ32ビット浮動小数点数に変換する
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int32_to_fp32(cnp.ndarray[cnp.int32_t, ndim = 1] a):
	cdef __m128i longa = _mm_set_epi32(a[3], a[2], a[1], a[0])
	cdef __m128 longb = _mm_cvtepi32_ps(longa)
	cdef cnp.ndarray[cnp.float32_t, ndim=1] result = np.empty(4)
	cdef float* pdata = <float*> result.data
	_mm_store_ps(pdata, longb)
	return result
