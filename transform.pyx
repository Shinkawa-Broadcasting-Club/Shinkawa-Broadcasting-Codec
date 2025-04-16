# cython: boundscheck=False, wraparound=False, nonecheck=False
cimport cython
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport rand, srand, malloc, free
from libc.time cimport time
from cython.parallel import parallel, prange
cdef extern from "nmmintrin.h" nogil:
	ctypedef struct __m128: pass
	__m128 _mm_loadu_ps (const float* mem_addr)
	__m128 _mm_set_ps1 (float a)
	__m128 _mm_add_ps (__m128 a, __m128 b)
	__m128 _mm_sub_ps (__m128 a, __m128 b)
	__m128 _mm_mul_ps (__m128 a, __m128 b)
	void _mm_storeu_ps (float* mem_addr, __m128 a)

cdef inline void dct_fwd(float[8] vector, float[8] out) nogil:
	cdef:
		float c4 = 0.70710677
		float c6 = 0.38268343
		float a2 = 0.5411961
		float a4 = 1.306563
		# stage 1
		float v0 = vector[0] + vector[7]
		float v1 = vector[1] + vector[6]
		float v2 = vector[2] + vector[5]
		float v3 = vector[3] + vector[4]
		float v4 = vector[3] - vector[4]
		float v5 = vector[2] - vector[5]
		float v6 = vector[1] - vector[6]
		float v7 = vector[0] - vector[7]
		# stage 2
		float v8 = v0 + v3
		float v9 = v1 + v2
		float v10 = v1 - v2
		float v11 = v0 - v3
		float v12 = v4 + v5
		float v13 = (v5 + v6) * c4
		float v14 = v6 + v7
		# stage 3
		float v17 = (v10 + v11) * c4
		float v18 = (v14 - v12) * c6
		# stage 4
		float v19 = v12 * a2 - v18
		float v20 = v14 * a4 - v18
		# stage 5
		float v23 = v13 + v7
		float v24 = v7 - v13
	out[0] = v8 + v9
	out[1] = v23 + v20
	out[2] = v17 + v11
	out[3] = v24 - v19
	out[4] = v8 - v9
	out[5] = v19 + v24
	out[6] = v11 - v17
	out[7] = v23 - v20

cdef inline void dct_time_fwd(float[8] inp, float[8] out) nogil:
	cdef:
		int i
	with nogil, parallel():
		for i in prange(8): out[i] = inp[i] + inp[i + 4] if i < 4 else inp[i - 4] - inp[i]
		for i in prange(8): out[i] = out[i] + out[i + 2] if i & 3 < 2 else out[i - 2] - out[i]
		for i in prange(8): out[i] = out[i] + out[i + 1] if i & 1 == 0 else out[i - 1] - out[i]

cpdef inline dct_3d_fwd(cnp.ndarray[cnp.float32_t, ndim=3] arr):
	return