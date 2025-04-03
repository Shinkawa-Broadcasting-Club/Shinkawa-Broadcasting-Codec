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
	void _mm_storeu_ps(float* mem_addr, __m128 a)
	void _MM_TRANSPOSE4_PS(__m128 row0, __m128 row1, __m128 row2, __m128 row3)
	__m128i _mm_loadl_epi64(const __m128i* mem_addr)
	__m128i _mm_cvtepu16_epi32(__m128i a)
	__m128 _mm_loadu_ps(const float* mem_addr)
	__m128 _mm_set_ps(float e3, float e2, float e1, float e0)
	__m128 _mm_add_ps(__m128 a, __m128 b)
	__m128 _mm_sub_ps(__m128 a, __m128 b)
	__m128 _mm_mul_ps(__m128 a, __m128 b)
	__m128 _mm_and_ps(__m128 a, __m128 b)

@cython.nogil
@cython.boundscheck(False)
@cython.wraparound(False)
cdef dct_time_fwd(cnp.ndarray[cnp.float32_t, ndim = 1] a):
	# dct_time_fwd(np.array(range(32)).astype(np.float32)) = [  28,  -4,  -8,   0, -16,   0,   0,   0,  92,  -4,  -8,   0, -16,   0,   0,   0, 156,  -4,  -8,   0, -16,   0,   0,   0, 220,  -4,  -8,   0, -16,   0,   0,   0]
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
cpdef dct_fwd(cnp.ndarray[cnp.float32_t, ndim = 1] a):
	# dct_fwd(np.array(range(32)).astype(np.float32)) = [28.0, -25.274143, 0.0, -2.2398288, 0.0, -0.44646287, 0.0, -0.039565086, 92.0, -25.274143, 0.0, -2.2398288, 0.0, -0.44646287, 0.0, -0.039565086, 156.0, -25.274143, 0.0, -2.2398288, 0.0, -0.44646287, 0.0, -0.039565086, 220.0, -25.274143, 0.0, -2.2398288, 0.0, -0.44646287, 0.0, -0.039565086]
	cdef:
		__m128 n0, n1, n2, n3, n4, n5, n6, n7, a1, a2, a3, a4, a5, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v17, v18, v19, v20, v23, v24, tmp
		float m0, m1, m2, m3, m4
		float* data = &a[0]
		cnp.ndarray[cnp.float32_t, ndim = 1] out = np.empty(32, dtype = np.float32)
	# define constants
	m0 = 0.7071067811865476
	m1 = -0.5411961001461969
	m2 = 1.3065629648763766
	m3 = 0.38268343236508984
	m4 = -1
	a1 = _mm_set_ps(m0, m0, m0, m0)
	a2 = _mm_set_ps(m1, m1, m1, m1)
	a3 = _mm_set_ps(m4, m4, m4, m4)
	a4 = _mm_set_ps(m2, m2, m2, m2)
	a5 = _mm_set_ps(m3, m3, m3, m3)
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
	# transform
	
	# store
	_MM_TRANSPOSE4_PS(n0, n2, n4, n6)
	_MM_TRANSPOSE4_PS(n1, n3, n5, n7)
	_mm_storeu_ps(&out[0], n0)
	_mm_storeu_ps(&out[4], n1)
	_mm_storeu_ps(&out[8], n2)
	_mm_storeu_ps(&out[12], n3)
	_mm_storeu_ps(&out[16], n4)
	_mm_storeu_ps(&out[20], n5)
	_mm_storeu_ps(&out[24], n6)
	_mm_storeu_ps(&out[28], n7)
	return out

@cython.nogil
@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint16_to_fp32(cnp.ndarray[cnp.uint16_t, ndim = 1] a):
	cdef:
		__m128i n0, n1
		cnp.uint16_t* data_ptr = &a[0]
	n1 = _mm_loadl_epi64((<__m128i *>data_ptr))
	n0 = _mm_cvtepu16_epi32(n1)
	return n0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dct_3d_fwd(cnp.ndarray[cnp.uint16_t, ndim = 3] a): # 32x32x32 block
	cdef:
		unsigned char i, j, k
		cnp.ndarray[cnp.uint16_t, ndim = 3] result = np.empty((32, 32, 32), np.float32)
	