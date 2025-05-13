# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False
from cython.parallel import prange, parallel
cimport numpy as cnp
import numpy as np
cdef extern from "<immintrin.h>" nogil:
    ctypedef struct __m128i: pass
    __m128i _mm_add_epi32(__m128i a, __m128i b)
    __m128i _mm_sub_epi32(__m128i a, __m128i b)
    __m128i _mm_hadd_epi32(__m128i a, __m128i b)
    __m128i _mm_hsub_epi32(__m128i a, __m128i b)
    __m128i _mm_sra_epi32(__m128i a, __m128i count)

cdef inline void dct_fwd_sse(__m128i[128] p, int s, int o) nogil:
    cdef __m128i[8] a, b
    # stage 1
    a[0] = _mm_add_epi32(p[o], p[(7 << s) + o])
    a[1] = _mm_add_epi32(p[(1 << s) + o], p[(6 << s) + o])
    a[2] = _mm_add_epi32(p[(2 << s) + o], p[(5 << s) + o])
    a[3] = _mm_add_epi32(p[(3 << s) + o], p[(4 << s) + o])
    a[4] = _mm_sub_epi32(p[o], p[(7 << s) + o])
    a[5] = _mm_sub_epi32(p[(1 << s) + o], p[(6 << s) + o])
    a[6] = _mm_sub_epi32(p[(2 << s) + o], p[(5 << s) + o])
    a[7] = _mm_sub_epi32(p[(3 << s) + o], p[(4 << s) + o])
    # stage 2
    b[0] = _mm_add_epi32(a[0], a[3])
    b[1] = _mm_add_epi32(a[1], a[2])
    b[2] = _mm_sub_epi32(a[0], a[3])
    b[3] = _mm_sub_epi32(a[1], a[2])
    b[4] = _mm_add_epi32(_mm_add_epi32(a[5], a[6]), _mm_add_epi32(_mm_sra_epi32(a[4], 1), a[4]))
    b[5] = _mm_sub_epi32(_mm_sub_epi32(a[4], a[7]), _mm_add_epi32(_mm_sra_epi32(a[6], 1), a[6]))
    b[6] = _mm_sub_epi32(_mm_add_epi32(a[4], a[7]), _mm_add_epi32(_mm_sra_epi32(a[5], 1), a[5]))
    b[7] = _mm_add_epi32(_mm_sub_epi32(a[5], a[6]), _mm_add_epi32(_mm_sra_epi32(a[7], 1), a[7]))
    # stage 3
    p[o] = _mm_add_epi32(b[0], b[1])
    p[(1 << s) + o] = _mm_add_epi32(b[4], _mm_sra_epi32(b[7], 2))
    p[(2 << s) + o] = _mm_add_epi32(_mm_sra_epi32(b[3], 1), b[2])
    p[(3 << s) + o] = _mm_add_epi32(_mm_sra_epi32(b[6], 2), b[5])
    p[(4 << s) + o] = _mm_sub_epi32(b[0], b[1])
    p[(5 << s) + o] = _mm_sub_epi32(b[6], _mm_sra_epi32(b[5], 2))
    p[(6 << s) + o] = _mm_sub_epi32(_mm_sra_epi32(b[2], 1), b[3])
    p[(7 << s) + o] = _mm_sub_epi32(_mm_sra_epi32(b[4], 2), b[7])

cdef inline void dct_bwd_sse(__m128i[128] p, int s, int o) nogil:
    cdef __m128i[8] a, b
    # stage 1
    a[0] = _mm_add_epi32(p[o], p[(4 << s) + o])
    a[1] = _mm_sub_epi32(p[o], p[(4 << s) + o])
    a[2] = _mm_sub_epi32(p[(6 << s) + o], _mm_sra_epi32(p[(2 << s) + o], 1))
    a[3] = _mm_add_epi32(p[(2 << s) + o], _mm_sra_epi32(p[(6 << s) + o], 1))
    a[4] = _mm_sub_epi32(_mm_sub_epi32(p[(5 << s) + o], p[(3 << s) + o]), _mm_add_epi32(p[(7 << s) + o], _mm_sra_epi32(p[(7 << s) + o], 1)))
    a[5] = _mm_sub_epi32(_mm_add_epi32(p[(7 << s) + o], p[(1 << s) + o]), _mm_add_epi32(p[(3 << s) + o], _mm_sra_epi32(p[(3 << s) + o], 1)))
    a[6] = _mm_add_epi32(_mm_sub_epi32(p[(7 << s) + o], p[(1 << s) + o]), _mm_add_epi32(p[(5 << s) + o], _mm_sra_epi32(p[(5 << s) + o], 1)))
    a[7] = _mm_add_epi32(_mm_add_epi32(p[(5 << s) + o], p[(3 << s) + o]), _mm_add_epi32(p[(1 << s) + o], _mm_sra_epi32(p[(1 << s) + o], 1)))
    # stage 2
    b[0] = _mm_add_epi32(a[0], a[3])
    b[1] = _mm_add_epi32(a[4], _mm_sra_epi32(a[7], 2))
    b[2] = _mm_sub_epi32(a[1], a[2])
    b[3] = _mm_add_epi32(a[5], _mm_sra_epi32(a[6], 2))
    b[4] = _mm_add_epi32(a[1], a[2])
    b[5] = _mm_sub_epi32(a[6], _mm_sra_epi32(a[5], 2))
    b[6] = _mm_sub_epi32(a[0], a[3])
    b[7] = _mm_sub_epi32(a[7], _mm_sra_epi32(a[4], 2))
    # stage 3
    p[o] = _mm_add_epi32(b[0], b[7])
    p[(1 << s) + o] = _mm_sub_epi32(b[2], b[5])
    p[(2 << s) + o] = _mm_add_epi32(b[4], b[3])
    p[(3 << s) + o] = _mm_add_epi32(b[6], b[1])
    p[(4 << s) + o] = _mm_sub_epi32(b[6], b[1])
    p[(5 << s) + o] = _mm_sub_epi32(b[4], b[3])
    p[(6 << s) + o] = _mm_add_epi32(b[2], b[5])
    p[(7 << s) + o] = _mm_sub_epi32(b[0], b[7])

cdef inline void dct_time_sse(__m128i[128] p, int o) nogil:
    cdef __m128i[2] a, b
    # stage 1
    a[0] = _mm_add_epi32(p[o << 1], p[(o << 1) + 1])
    a[1] = _mm_sub_epi32(p[o << 1], p[(o << 1) + 1])
    # stage 2
    b[0] = _mm_hadd_epi32(a[0], a[1])
    b[1] = _mm_hsub_epi32(a[0], a[1])
    # stage 3
    p[o << 1] = _mm_hadd_epi32(b[0], b[1])
    p[(o << 1) + 1] = _mm_hsub_epi32(b[0], b[1])

cdef inline void dct3d_sse(__m128i[128] i) nogil:
    cdef int j, k
    for j in range(32):
        for k in range(2): dct_fwd_sse(i, 1, (j << 1) + k)
    for j in range(4):
        for k in range(16): dct_fwd_sse(i, 4, (j << 4) + k)
    for j in range(64): dct_fwd_sse(i, j << 1)

cdef inline void idct3d_sse(__m128i[128] i) nogil:
    cdef int j, k
    for j in range(64): dct_fwd_sse(i, j << 1)
    for j in range(4):
        for k in range(16): dct_bwd_sse(i, 4, (j << 4) + k)
    for j in range(32):
        for k in range(2): dct_bwd_sse(i, 1, (j << 1) + k)
