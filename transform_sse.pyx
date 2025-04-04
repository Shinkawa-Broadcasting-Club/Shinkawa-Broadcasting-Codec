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
cdef ndarray_to_c_array(cnp.ndarray[cnp.uint16_t, ndim = 3] a):
	if not a.flags['C_CONTIGUOUS']: a = np.ascontiguousarray(a)
	return a

@cython.nogil
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _mm_lval_ps(float a):
	cdef __m128 result = _mm_set_ps(a, a, a, a)
	return result

@cython.nogil
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _mm_ldarr_ps(cnp.ndarray[cnp.float32_t, ndim = 1] a):
	cdef:
		__m128 n0, n1, n2, n3, n4, n5, n6, n7, v0, v1, v2, v3, v4, v5, v6, v7
		float* data = &a[0]
	v0 = _mm_loadu_ps(data)
	v1 = _mm_loadu_ps(data + 4)
	v2 = _mm_loadu_ps(data + 8)
	v3 = _mm_loadu_ps(data + 12)
	v4 = _mm_loadu_ps(data + 16)
	v5 = _mm_loadu_ps(data + 20)
	v6 = _mm_loadu_ps(data + 24)
	v7 = _mm_loadu_ps(data + 28)
	_MM_TRANSPOSE4_PS(n0, n2, n4, n6)
	_MM_TRANSPOSE4_PS(n1, n3, n5, n7)
	n0 = v0
	n1 = v2
	n2 = v4
	n3 = v6
	n4 = v1
	n5 = v3
	n6 = v5
	n7 = v7
	return v0, v1, v2, v3, v4, v5, v6, v7

@cython.nogil
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _mm_starr_ps(__m128 v0, __m128 v1, __m128 v2, __m128 v3, __m128 v4, __m128 v5, __m128 v6, __m128 v7):
	cdef:
		__m128 n0, n1, n2, n3, n4, n5, n6, n7
		cnp.ndarray[cnp.float32_t, ndim = 1] out = np.empty(32, dtype = np.float32)
	v0 = n0
	v1 = n4
	v2 = n1
	v3 = n5
	v4 = n2
	v5 = n6
	v6 = n3
	v7 = n7
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
cdef dct_time_fwd(cnp.ndarray[cnp.float32_t, ndim = 1] a):
	# dct_time_fwd(np.array(range(32)).astype(np.float32)) = [  28,  -4,  -8,   0, -16,   0,   0,   0,  92,  -4,  -8,   0, -16,   0,   0,   0, 156,  -4,  -8,   0, -16,   0,   0,   0, 220,  -4,  -8,   0, -16,   0,   0,   0]
	cdef:
		__m128 n0, n1, n2, n3, n4, n5, n6, n7, v0, v1, v2, v3, v4, v5, v6, v7
		cnp.ndarray[cnp.float32_t, ndim = 1] out = np.empty(32, dtype = np.float32)
		float* data = &a[0]
	# load
	v0 = _mm_loadu_ps(data)
	v1 = _mm_loadu_ps(data + 4)
	v2 = _mm_loadu_ps(data + 8)
	v3 = _mm_loadu_ps(data + 12)
	v4 = _mm_loadu_ps(data + 16)
	v5 = _mm_loadu_ps(data + 20)
	v6 = _mm_loadu_ps(data + 24)
	v7 = _mm_loadu_ps(data + 28)
	_MM_TRANSPOSE4_PS(n0, n2, n4, n6)
	_MM_TRANSPOSE4_PS(n1, n3, n5, n7)
	n0 = v0
	n1 = v2
	n2 = v4
	n3 = v6
	n4 = v1
	n5 = v3
	n6 = v5
	n7 = v7
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
	# store
	v0 = n0
	v1 = n4
	v2 = n1
	v3 = n5
	v4 = n2
	v5 = n6
	v6 = n3
	v7 = n7
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
cpdef dct_fwd(cnp.ndarray[cnp.float32_t, ndim = 1] a):
	# dct_fwd(np.array(range(32)).astype(np.float32)) = [28.0, -25.274143, 0.0, -2.2398288, 0.0, -0.44646287, 0.0, -0.039565086, 92.0, -25.274143, 0.0, -2.2398288, 0.0, -0.44646287, 0.0, -0.039565086, 156.0, -25.274143, 0.0, -2.2398288, 0.0, -0.44646287, 0.0, -0.039565086, 220.0, -25.274143, 0.0, -2.2398288, 0.0, -0.44646287, 0.0, -0.039565086]
	cdef:
		__m128 n0, n1, n2, n3, n4, n5, n6, n7, v0, v1, v2, v3, v4, v5, v6, v7, a1, a2, a3, a4, a5, tmp
		float m0, m1, m2, m3, m4
		cnp.ndarray[cnp.float32_t, ndim = 1] out = np.empty(32, dtype = np.float32)
		float* data = &a[0]
	# load
	v0 = _mm_loadu_ps(data)
	v1 = _mm_loadu_ps(data + 4)
	v2 = _mm_loadu_ps(data + 8)
	v3 = _mm_loadu_ps(data + 12)
	v4 = _mm_loadu_ps(data + 16)
	v5 = _mm_loadu_ps(data + 20)
	v6 = _mm_loadu_ps(data + 24)
	v7 = _mm_loadu_ps(data + 28)
	_MM_TRANSPOSE4_PS(n0, n2, n4, n6)
	_MM_TRANSPOSE4_PS(n1, n3, n5, n7)
	n0 = v0
	n1 = v2
	n2 = v4
	n3 = v6
	n4 = v1
	n5 = v3
	n6 = v5
	n7 = v7
	# stage 1
	v0 = _mm_add_ps(n0, n7)
	v1 = _mm_add_ps(n1, n6)
	v2 = _mm_add_ps(n2, n5)
	v3 = _mm_add_ps(n3, n4)
	v4 = _mm_sub_ps(n3, n4)
	v5 = _mm_sub_ps(n2, n5)
	v6 = _mm_sub_ps(n1, n6)
	v7 = _mm_sub_ps(n0, n7)
	# stage 2
	n0 = _mm_add_ps(v0, v3)
	n1 = _mm_add_ps(v1, v2)
	n2 = _mm_sub_ps(v1, v2)
	n3 = _mm_sub_ps(v0, v3)
	n4 = _mm_add_ps(v4, v5)
	n5 = _mm_add_ps(v5, v6)
	n6 = _mm_add_ps(v6, v7)
	n7 = v7
	# stage 3
	v0 = _mm_add_ps(n0, n1)
	v1 = _mm_sub_ps(n0, n1)
	v2 = _mm_add_ps(n2, n3)
	v3 = n3
	v4 = n4
	v5 = n5
	v6 = n6
	v7 = n7
	tmp = _mm_sub_ps(n6, n4)
	# stage 4
	n0 = v0
	n1 = v1
	n2 = _mm_mul_ps(v2, a1)
	n3 = v3
	n4 = _mm_mul_ps(v4, a2)
	n5 = _mm_mul_ps(v5, a3)
	n6 = _mm_mul_ps(v6, a4)
	tmp = _mm_mul_ps(tmp, a5)
	# stage 5
	v0 = n0
	v1 = n1
	v2 = _mm_add_ps(n3, n2)
	v3 = _mm_sub_ps(n3, n2)
	v4 = _mm_sub_ps(n4, tmp)
	v5 = _mm_add_ps(n7, n5)
	v6 = _mm_sub_ps(n6, tmp)
	v7 = _mm_sub_ps(n7, n5)
	# stage 6
	n0 = v0
	n1 = v1
	n2 = v2
	n3 = v3
	n4 = _mm_add_ps(v7, v4)
	n5 = _mm_add_ps(v5, v6)
	n6 = _mm_sub_ps(v5, v6)
	n7= _mm_sub_ps(v7, v4)
	# stage 7
	v0 = n0
	v4 = n1
	v2 = n2
	v6 = n3
	v5 = n4
	v1 = n5
	v7 = n6
	v3 = n7
	# store
	n0 = v0
	n1 = v4
	n2 = v1
	n3 = v5
	n4 = v2
	n5 = v6
	n6 = v3
	n7 = v7
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
cpdef uint16_to_fp32(cnp.ndarray[cnp.uint16_t, ndim = 1] a):
	cdef:
		__m128i n0, n1
		cnp.uint16_t* data_ptr = &a[0]
	n1 = _mm_loadl_epi64((<__m128i *>data_ptr))
	n0 = _mm_cvtepu16_epi32(n1)
	return n0

@cython.nogil
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dct_3d_fwd(short[:, :, :] arr): # 32x32x32 block
	cdef:
		cnp.ndarray[cnp.uint16_t, ndim = 3] result = np.empty((32, 32, 32), np.float32)
		float[:, :, :] tmp = result
		unsigned char h, i, j, k, l, m
	for i in prange(32, nogil = True):
		for j in prange(32, nogil = True):
			for k in prange(8, nogil = True):
				l = k << 2
				m = l + 4
				tmp[i, j, l:m] = uint16_to_fp32(np.asarray(arr[i, j, l:m]))
			tmp[:, i, j] = dct_time_fwd(tmp[:, i, j])
	for i in prange(32, nogil = True):
		for j in prange(32, nogil = True):
			tmp[j, :, i] = dct_fwd(tmp[j, :, i])
	for i in prange(32, nogil = True):
		for j in prange(32, nogil = True):
			tmp[i, j, :] = dct_fwd(tmp[i, j, :])
	return np.asarray(tmp)