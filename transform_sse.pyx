import numpy as np
cimport cython
cimport numpy as cnp
from libc.string cimport memcpy
from cython.parallel import prange
cdef extern from *:
	"""
    #include <nmmintrin.h>
	"""
	ctypedef struct __m128i: pass # 32ビット整数 x 4
	ctypedef struct __m128:	pass # 32ビット浮動小数点数 x 4
	ctypedef struct __m64: pass
	__m64 _mm_set_pi16(short e3, short e2, short e1, short e0)
	__m128i _mm_loadu_si64(const void *mem_addr)
	__m128i _mm_cvtepu16_epi32(__m128i a)
	__m128i _mm_set_epi16(short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
	__m128i _mm_set_epi32(int e3, int e2, int e1, int e0)
	__m128i _mm_slli_epi32(__m128i a, int imm8)
	__m128i _mm_and_si128(__m128i a, __m128i b)
	__m128i _mm_castps_si128(__m128 a)
	__m128 _mm_loadu_ps(const float *mem_addr)
	void _mm_storeu_ps(float* mem_addr, __m128 a)
	void _MM_TRANSPOSE4_PS (__m128 row0, __m128 row1, __m128 row2, __m128 row3)
	__m128 _mm_cvtpu16_ps(__m64 a)
	__m128 _mm_castsi128_ps(__m128i a)
	__m128 _mm_add_ps(__m128 a, __m128 b)
	__m128 _mm_sub_ps(__m128 a, __m128 b)
	__m128 _mm_mul_ps(__m128 a, __m128 b)

@cython.nogil
@cython.boundscheck(False)
@cython.wraparound(False)
cdef dct_time_fwd(cnp.ndarray[cnp.float32_t, ndim = 1] a):
	cdef:
		__m128 n0, n1, n2, n3, n4, n5, n6, n7, v0, v1, v2, v3, v4, v5, v6, v7
		float* data = &a[0]
		cnp.ndarray[cnp.float32_t, ndim = 1] out = np.empty(32, dtype = np.float32)
	# load
	n0 = _mm_loadu_ps(data)
	n1 = _mm_loadu_ps(data + 4)
	n2 = _mm_loadu_ps(data + 8)
	n3 = _mm_loadu_ps(data + 12)
	n4 = _mm_loadu_ps(data + 16)
	n5 = _mm_loadu_ps(data + 20)
	n6 = _mm_loadu_ps(data + 24)
	n7 = _mm_loadu_ps(data + 28)
	_MM_TRANSPOSE4_PS(n0, n2, n4, n6)
	_MM_TRANSPOSE4_PS(n1, n3, n5, n7)
	# stage 1
	v0 = _mm_add_ps(n0, n4)
	v1 = _mm_add_ps(n1, n5)
	v2 = _mm_add_ps(n2, n6)
	v3 = _mm_add_ps(n3, n7)
	v4 = _mm_sub_ps(n0, n4)
	v5 = _mm_sub_ps(n1, n5)
	v6 = _mm_sub_ps(n2, n6)
	v7 = _mm_sub_ps(n3, n7)
	# stage 2
	n0 = _mm_add_ps(v0, v2)
	n1 = _mm_add_ps(v1, v3)
	n2 = _mm_sub_ps(v0, v2)
	n3 = _mm_sub_ps(v1, v3)
	n4 = _mm_add_ps(v4, v6)
	n5 = _mm_add_ps(v5, v7)
	n6 = _mm_sub_ps(v4, v6)
	n7 = _mm_sub_ps(v5, v7)
	# stage 3
	v0 = _mm_add_ps(n0, n1)
	v1 = _mm_sub_ps(n0, n1)
	v2 = _mm_add_ps(n2, n3)
	v3 = _mm_sub_ps(n2, n3)
	v4 = _mm_add_ps(n4, n5)
	v5 = _mm_sub_ps(n4, n5)
	v6 = _mm_add_ps(n6, n7)
	v7 = _mm_sub_ps(n6, n7)
	# set to ndarray
	_MM_TRANSPOSE4_PS(v0, v2, v4, v6)
	_MM_TRANSPOSE4_PS(v1, v3, v5, v7)
	_mm_storeu_ps(&out[0], v0)
	_mm_storeu_ps(&out[4], v1)
	_mm_storeu_ps(&out[8], v2)
	_mm_storeu_ps(&out[12], v3)
	_mm_storeu_ps(&out[16], v4)
	_mm_storeu_ps(&out[20], v5)
	_mm_storeu_ps(&out[24], v6)
	_mm_storeu_ps(&out[28], v7)
	return out

@cython.boundscheck(False)
@cython.wraparound(False)
def dct_fwd(cnp.ndarray[cnp.float32_t, ndim = 1] a):
	cdef:
		__m128 n0, n1, n2, n3, n4, n5, n6, n7, v0, v1, v2, v3, v4, v5, v6, v7
		float* data = &a[0]
		cnp.ndarray[cnp.float32_t, ndim = 1] out = np.empty(32, dtype = np.float32)
	# load
	n0 = _mm_loadu_ps(data)
	n1 = _mm_loadu_ps(data + 4)
	n2 = _mm_loadu_ps(data + 8)
	n3 = _mm_loadu_ps(data + 12)
	n4 = _mm_loadu_ps(data + 16)
	n5 = _mm_loadu_ps(data + 20)
	n6 = _mm_loadu_ps(data + 24)
	n7 = _mm_loadu_ps(data + 28)
	_MM_TRANSPOSE4_PS(n0, n2, n4, n6)
	_MM_TRANSPOSE4_PS(n1, n3, n5, n7)

	_MM_TRANSPOSE4_PS(v0, v2, v4, v6)
	_MM_TRANSPOSE4_PS(v1, v3, v5, v7)
	_mm_storeu_ps(&out[0], v0)
	_mm_storeu_ps(&out[4], v1)
	_mm_storeu_ps(&out[8], v2)
	_mm_storeu_ps(&out[12], v3)
	_mm_storeu_ps(&out[16], v4)
	_mm_storeu_ps(&out[20], v5)
	_mm_storeu_ps(&out[24], v6)
	_mm_storeu_ps(&out[28], v7)
	return out

@cython.nogil
@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint16_to_fp32(cnp.ndarray[cnp.uint16_t, ndim = 1] a):
	cdef:
		unsigned char* data_ptr = <unsigned char *> ...
		__m128i n0 = _mm_cvtepu16_epi32(_mm_loadu_si64(data_ptr))
	return n0

@cython.boundscheck(False)
@cython.wraparound(False)
def dct_3d_fwd(cnp.ndarray[cnp.uint16_t, ndim = 3] a): # 32x32x32 block
	cdef:
		unsigned char i, j, k
		cnp.ndarray[cnp.uint16_t, ndim = 3] result = np.empty((32, 32, 32), np.float32)
	