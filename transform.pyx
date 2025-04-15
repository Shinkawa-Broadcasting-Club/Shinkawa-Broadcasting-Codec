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

cdef inline dct_fwd(__m128[8] arr_in):
    cdef:
        __m128[8] arr_out
        __m128[7] s
        __m128 v0, v1, v2, v3, v4, v5, v6, v7  # stage 1
        __m128 v8, v9, v10, v11, v12, v13, v14 # stage 2
        __m128 v17, v18              # stage 3
        __m128 v19, v20, v23, v24    # stage 4
        __m128 a2 = _mm_set_ps1(0x3F0A8BD4)
        __m128 a4 = _mm_set_ps1(0x3FA73D75)
        __m128 c4 = _mm_set_ps1(0x3F3504F3)
        __m128 c6 = _mm_set_ps1(0x3EC3EF15)
        int i
    with nogil, parallel():
        for i in prange(8):
            if i == 0:   v0 = _mm_add_ps(arr_in[0], arr_in[7])
            elif i == 1: v1 = _mm_add_ps(arr_in[1], arr_in[6])
            elif i == 2: v2 = _mm_add_ps(arr_in[2], arr_in[5])
            elif i == 3: v3 = _mm_add_ps(arr_in[3], arr_in[4])
            elif i == 4: v4 = _mm_sub_ps(arr_in[3], arr_in[4])
            elif i == 5: v5 = _mm_sub_ps(arr_in[2], arr_in[5])
            elif i == 6: v6 = _mm_sub_ps(arr_in[1], arr_in[6])
            elif i == 7: v7 = _mm_sub_ps(arr_in[0], arr_in[7])
        for i in prange(7):
            if i == 0:   v8 = _mm_add_ps(v0, v3)
            elif i == 1: v9 = _mm_add_ps(v1, v2)
            elif i == 2: v10 = _mm_sub_ps(v1, v2)
            elif i == 3: v11 = _mm_sub_ps(v0, v3)
            elif i == 4: v12 = _mm_add_ps(v4, v5)
            elif i == 5: v13 = _mm_mul_ps(_mm_add_ps(v5 + v6), c4)
            elif i == 6: v14 = _mm_add_ps(v6, v7)
        for i in prange(4):
            if i == 0:   arr_out[0] = _mm_add_ps(v8, v9)
            elif i == 1: arr_out[4] = _mm_sub_ps(v8, v9)
            elif i == 2: v17 = _mm_mul_ps(_mm_add_ps(v10, v11), c4)
            elif i == 3: v18 = _mm_mul_ps(_mm_sub_ps(v14, v12), c6)
        for i in prange(6):
            if i == 0:   v19 = _mm_sub_ps(_mm_mul_ps(v12, a2), v18)
            elif i == 1: v20 = _mm_sub_ps(_mm_mul_ps(v14, a4), v18)
            elif i == 2: arr_out[2] = _mm_add_ps(v11, v17)
            elif i == 3: arr_out[6] = _mm_sub_ps(v11, v17)
            elif i == 4: v23 = _mm_add_ps(v7, v13)
            elif i == 5: v24 = _mm_sub_ps(v7, v13)
        for i in prange(4):
            if i == 0:   arr_out[5] = _mm_add_ps(v24, v19)
            elif i == 1: arr_out[1] = _mm_add_ps(v23, v20)
            elif i == 2: arr_out[7] = _mm_sub_ps(v23, v20)
            elif i == 3: arr_out[3] = _mm_sub_ps(v24, v19)
    return arr_out

cdef inline dct_3d_fwd(float[8][8][8] arr):
    cdef:
        float[8][8][8] out
        __m128[8][8][2] tmp
        int i, j, k
    with nogil:
        for i in prange(8):
            for j in prange(8):
                for k in prange(2):
                    tmp[i][j][k] = _mm_loadu_ps(&arr[i][j][k << 2])
        for i in prange(8):
            for j in prange(8):
                for k in prange(2):
                    _mm_storeu_ps(&arr[i][j][k << 2], tmp[i][j][k])        