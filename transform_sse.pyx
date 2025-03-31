import numpy as np
import cython
cimport cython
cimport numpy as cnp
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
	__m128i _mm_set_epi32(int e3, int e2, int e1, int e0)
	__m128i _mm_shuffle_epi32(__m128i a, int imm8)
	__m128i _mm_add_epi32(__m128i a, __m128i b)
	__m128i _mm_hadd_epi32(__m128i a, __m128i b)
	__m128i _mm_sub_epi32(__m128i a, __m128i b)
	__m128i _mm_hsub_epi32(__m128i a, __m128i b)
	__m128i _mm_castpd_si128(__m128d a)
	__m128d _mm_move_sd (__m128d a, __m128d b)
	__m128d _mm_castsi128_pd(__m128i a)
	__m128 _mm_cvtepi32_ps(__m128i a)
	void _mm_store_si128(__m128i* p, __m128i a)
	__m128 _mm_store_ps(float *p, __m128 a)

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
	cdef __m128i n4 = _mm_hadd_epi32(n2, n3)
	cdef __m128i n5 = _mm_hsub_epi32(n2, n3)
    # stage 3
	cdef __m128i n6 = _mm_hadd_epi32(n4, n5)
	cdef __m128i n7 = _mm_hsub_epi32(n4, n5)
    # ordering
	cdef __m128i n8 = _mm_shuffle_epi32(n6, 141)
	cdef __m128i n9 = _mm_shuffle_epi32(n7, 216)
	# casting to inter-register ordering
	cdef __m128d n10 = _mm_castsi128_pd(n8)
	cdef __m128d n11 = _mm_castsi128_pd(n9)
	# inter-register ordering
	cdef __m128d n12 = _mm_move_sd(n10, n11)
	cdef __m128d n13 = _mm_move_sd(n11, n10)
	# casting to integer
	cdef __m128i n14 = _mm_castpd_si128(n12)
	cdef __m128i n15 = _mm_castpd_si128(n13)
	# re-ordering
	cdef __m128i n16 = _mm_shuffle_epi32(n14, 78)
	cdef cnp.ndarray[cnp.int32_t, ndim=1] result = np.empty(8, dtype = np.int32)
	cdef cnp.ndarray[cnp.int32_t, ndim=1] tmp0 = np.empty(4, dtype = np.int32)
	cdef cnp.ndarray[cnp.int32_t, ndim=1] tmp1 = np.empty(4, dtype = np.int32)
	cdef int* tdata0 = <int*> tmp0.data
	cdef int* tdata1 = <int*> tmp1.data
	_mm_store_si128(<__m128i*> tdata0, n16)
	_mm_store_si128(<__m128i*> tdata1, n15)
	result[:4] = tmp0
	result[4:] = tmp1
	return result