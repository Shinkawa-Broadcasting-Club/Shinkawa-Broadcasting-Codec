import numpy as np
cimport cython
cimport numpy as cnp
from libc.string cimport memcpy
cdef extern from *:
	"""
	#include <xmmintrin.h>
	#include <emmintrin.h>
    #include <tmmintrin.h>
	"""
	ctypedef struct __m128d: pass # 64ビット浮動小数点数 x 2
	ctypedef struct __m128i: pass # 32ビット整数 x 4
	ctypedef struct __m128:	pass # 32ビット浮動小数点数 x 4
	__m128i _mm_set_epi16(short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
	__m128i _mm_add_epi32(__m128i a, __m128i b)
	__m128i _mm_hadd_epi32(__m128i a, __m128i b)
	__m128i _mm_sub_epi32(__m128i a, __m128i b)
	__m128i _mm_hsub_epi32(__m128i a, __m128i b)
	__m128i _mm_castps_si128 (__m128 a)
	__m128 _mm_castsi128_ps(__m128i a)
	__m128 _mm_shuffle_ps (__m128 a, __m128 b, unsigned int imm8)

@cython.boundscheck(False)
@cython.wraparound(False)
def dct_time_fwd(cnp.ndarray[cnp.uint16_t, ndim = 1] a):
	cdef unsigned short s = 0
	# convert uint16 to int32
	cdef __m128i n0 = _mm_set_epi16(s, a[3], s, a[2], s, a[1], s, a[0])
	cdef __m128i n1 = _mm_set_epi16(s, a[7], s, a[6], s, a[5], s, a[4])
	# stage 1
	cdef __m128i n2 = _mm_add_epi32(n0, n1)
	cdef __m128i n3 = _mm_sub_epi32(n0, n1)
	# stage 2
	n0 = _mm_hadd_epi32(n2, n3)
	n1 = _mm_hsub_epi32(n2, n3)
	# stage 3
	n2 = _mm_hadd_epi32(n0, n1)
	n3 = _mm_hsub_epi32(n0, n1)
	# ordering[(0, 4, 1, 5, 2, 6, 3, 7) => (0, 1, 2, 3, 4, 5, 6, 7)]
	cdef __m128 n4 = _mm_castsi128_ps(n2)
	cdef __m128 n5 = _mm_castsi128_ps(n3)
	cdef __m128 n6 = _mm_shuffle_ps(n4, n5, 136)
	cdef __m128 n7 = _mm_shuffle_ps(n4, n5, 221)
	cdef cnp.ndarray[cnp.int32_t, ndim=1] result = np.empty(8, dtype=np.int32)
	memcpy(&result[0], &n6, 16)
	memcpy(&result[4], &n7, 16)
	return result

