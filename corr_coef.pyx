cimport cython
from libc.stdlib cimport rand, srand
from libc.time cimport time
from cython.parallel import prange
cdef extern from "nmmintrin.h":
    ctypedef struct __m128: pass
    __m128 _mm_loadu_ps (const float* mem_addr)
    __m128 _mm_add_ps(__m128 a, __m128 b)
    void _mm_storeu_ps (float* mem_addr, __m128 a)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef rand256(float[:] arr):
    cdef:
        int N = arr.shape[0], i, j
        float[256] out
    if N < 256: raise ValueError("配列の要素数は最低でも256以上必要です")
    srand(time(NULL))
    for i in range(256): out[i] = arr[i]
    for i in range(256, N):
        j = rand() % (i + 1)
        if j < 256: out[j] = arr[i]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef corr_coef(float[:] x, float[:] y):
    cdef:
        float[256] sample_x = rand256(x), sample_y = rand256(y), dev_x, dev_y
        __m128[64] tmp0_x, tmp0_y, tmp1_x, tmp1_y
        float[4] tmp_x, tmp_y
        float avg_x, avg_y
        unsigned short i, j, k
    for i in prange(64, nogil = True, schedule = 'static'):
        j = i << 2
        for k in prange(2, nogil = True, schedule = 'static'):
            if k == 0: tmp0_x[i] = _mm_loadu_ps(&sample_x[j])
            if k == 1: tmp0_y[i] = _mm_loadu_ps(&sample_y[j])
    for i in prange(32, nogil = True, schedule = 'static'):
        j = i << 1
        for k in prange(2, nogil = True, schedule = 'static'):
            if k == 0: tmp1_x[i] = _mm_add_ps(tmp0_x[j], tmp0_x[j + 1])
            if k == 1: tmp1_y[i] = _mm_add_ps(tmp0_y[j], tmp0_y[j + 1])
    for i in prange(16, nogil = True, schedule = 'static'):
        j = i << 1
        for k in prange(2, nogil = True, schedule = 'static'):
            if k == 0: tmp0_x[i] = _mm_add_ps(tmp1_x[j], tmp1_x[j + 1])
            if k == 1: tmp0_y[i] = _mm_add_ps(tmp1_y[j], tmp1_y[j + 1])
    for i in prange(8, nogil = True, schedule = 'static'):
        j = i << 1
        for k in prange(2, nogil = True, schedule = 'static'):
            if k == 0: tmp1_x[i] = _mm_add_ps(tmp0_x[j], tmp0_x[j + 1])
            if k == 1: tmp1_y[i] = _mm_add_ps(tmp0_y[j], tmp0_y[j + 1])
    for i in prange(4, nogil = True, schedule = 'static'):
        j = i << 1
        for k in prange(2, nogil = True, schedule = 'static'):
            if k == 0: tmp0_x[i] = _mm_add_ps(tmp1_x[j], tmp1_x[j + 1])
            if k == 1: tmp0_y[i] = _mm_add_ps(tmp1_y[j], tmp1_y[j + 1])
    for i in prange(2, nogil = True, schedule = 'static'):
        j = i << 1
        for k in prange(2, nogil = True, schedule = 'static'):
            if k == 0: tmp1_x[i] = _mm_add_ps(tmp0_x[j], tmp0_x[j + 1])
            if k == 1: tmp1_y[i] = _mm_add_ps(tmp0_y[j], tmp0_y[j + 1])
    for k in prange(2, nogil = True, schedule = 'static'):
        if k == 0:
            _mm_storeu_ps(&tmp_x[0], _mm_add_ps(tmp1_x[0], tmp1_x[1]))
            for i in prange(2, nogil = True, schedule = 'static'):
                if i == 0: tmp_x[0] += tmp_x[1]
                if i == 1: tmp_x[2] += tmp_x[3]
        if k == 1:
            _mm_storeu_ps(&tmp_y[0], _mm_add_ps(tmp1_y[0], tmp1_y[1]))
            for i in prange(2, nogil = True, schedule = 'static'):
                if i == 0: tmp_y[0] += tmp_y[1]
                if i == 1: tmp_y[2] += tmp_y[3]
    avg_x = tmp_x[0] + tmp_x[2]
    avg_y = tmp_y[0] + tmp_y[2]
    