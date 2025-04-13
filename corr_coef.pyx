cimport cython
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport rand, srand, malloc, free
from libc.time cimport time
from cython.parallel import prange
cdef extern from "nmmintrin.h":
    ctypedef struct __m128: pass
    __m128 _mm_load_ss (const float* mem_addr)
    __m128 _mm_loadu_ps (const float* mem_addr)
    __m128 _mm_rsqrt_ss (__m128 a)
    __m128 _mm_add_ps (__m128 a, __m128 b)
    __m128 _mm_hadd_ps (__m128 a, __m128 b)
    __m128 _mm_mul_ps (__m128 a, __m128 b)
    void _mm_store_ss (float* mem_addr, __m128 a)
    void _mm_storeu_ps (float* mem_addr, __m128 a)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline rand256(float[:, :] arr):
    cdef:
        int N = arr.shape[1]
        float[2][256] out
        int* indices = <int*> malloc(N * sizeof(int))
        int i, j, k, tmp
    if N < 256: raise ValueError("入力配列の第2軸（列数）は256以上である必要があります。")
    if indices == NULL: raise MemoryError("インデックス配列のメモリ確保に失敗しました。")
    for i in range(N): indices[i] = i
    srand(<unsigned int> time(NULL))
    for i in range(256):
        j = i + rand() % (N - i)
        tmp = indices[i]
        indices[i] = indices[j]
        indices[j] = tmp
        for k in range(2): out[k][i] = arr[k, indices[i]]
    free(indices)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline rsqrt(float n):
    cdef float d
    _mm_store_ss(&d, _mm_rsqrt_ss(_mm_load_ss(&n)))
    return d

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline corr_coef(float[:, :] arr):
    # float[0]: xの値
    # float[1]: yの値
    #   sample: 無作為抽出したサンプル
    #      avg: 平均
    #      dev: 偏差
    #      var: dev²の合計
    #      cov: dev[0] * dev[1]の合計
    cdef:
        float[2][256] sample = rand256(arr)
        float[256] x, y, xx, yy, xy
        float cov, corr
        unsigned short i, j, k, l
    with nogil:
        for i in prange(5):
            for j in prange(256):
                if i == 0: x[j] = sample[0][j]
                if i == 1: y[j] = sample[1][j]
                if i == 2: xx[j] = sample[0][j] ** 2
                if i == 3: yy[j] = sample[1][j] ** 2
                if i == 4: xy[j] = sample[0][j] * sample[1][j]
        for i in prange(5):
            for j in prange(128):
                k = j << 1
                l = k + 1
                if i == 0: x[k] += x[l]
                if i == 1: y[k] += y[l]
                if i == 2: xx[k] += xx[l]
                if i == 3: yy[k] += yy[l]
                if i == 4: xy[k] += xy[l]
        for i in prange(5):
            for j in prange(64):
                k = j << 2
                l = k + 2
                if i == 0: x[k] += x[l]
                if i == 1: y[k] += y[l]
                if i == 2: xx[k] += xx[l]
                if i == 3: yy[k] += yy[l]
                if i == 4: xy[k] += xy[l]
        for i in prange(5):
            for j in prange(32):
                k = j << 3
                l = k + 4
                if i == 0: x[k] += x[l]
                if i == 1: y[k] += y[l]
                if i == 2: xx[k] += xx[l]
                if i == 3: yy[k] += yy[l]
                if i == 4: xy[k] += xy[l]
        for i in prange(5):
            for j in prange(16):
                k = j << 4
                l = k + 8
                if i == 0: x[k] += x[l]
                if i == 1: y[k] += y[l]
                if i == 2: xx[k] += xx[l]
                if i == 3: yy[k] += yy[l]
                if i == 4: xy[k] += xy[l]
        for i in prange(5):
            for j in prange(8):
                k = j << 5
                l = k + 16
                if i == 0: x[k] += x[l]
                if i == 1: y[k] += y[l]
                if i == 2: xx[k] += xx[l]
                if i == 3: yy[k] += yy[l]
                if i == 4: xy[k] += xy[l]
        for i in prange(5):
            for j in prange(4):
                k = j << 6
                l = k + 32
                if i == 0: x[k] += x[l]
                if i == 1: y[k] += y[l]
                if i == 2: xx[k] += xx[l]
                if i == 3: yy[k] += yy[l]
                if i == 4: xy[k] += xy[l]
        for i in prange(5):
            for j in prange(2):
                k = j << 7
                l = k + 64
                if i == 0: x[k] += x[l]
                if i == 1: y[k] += y[l]
                if i == 2: xx[k] += xx[l]
                if i == 3: yy[k] += yy[l]
                if i == 4: xy[k] += xy[l]
        for i in prange(5):
            if i == 0: x[0] += x[128]
            if i == 1: y[0] += y[128]
            if i == 2: xx[0] += xx[128]
            if i == 3: yy[0] += yy[128]
            if i == 4: xy[0] += xy[128]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline corr(cnp.ndarray[cnp.float32_t, ndim=2] arr):
    cdef float[:, :] n = arr
    return corr_coef(n)