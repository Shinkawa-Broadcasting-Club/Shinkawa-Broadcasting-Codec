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
cdef inline fsum(float[256] arr):
    cdef:
        __m128[64] tmp
        float[4] dst
    for i in prange(64):       tmp[i] = _mm_loadu_ps(&arr[i << 2])
    for i in prange(0, 32, 2): tmp[i] = _mm_add_ps(tmp[i], tmp[i + 1])
    for i in prange(0, 16, 4): tmp[i] = _mm_add_ps(tmp[i], tmp[i + 2])
    for i in prange(0, 8, 8):  tmp[i] = _mm_add_ps(tmp[i], tmp[i + 4])
    for i in prange(0, 4, 16): tmp[i] = _mm_add_ps(tmp[i], tmp[i + 8])
    for i in prange(0, 2, 32): tmp[i] = _mm_add_ps(tmp[i], tmp[i + 16])
    tmp[0] = _mm_add_ps(tmp[0], tmp[32])
    for i in range(2): tmp[0] = _mm_hadd_ps(tmp[0], tmp[0])
    _mm_storeu_ps(&dst[0], tmp[0])
    return dst[0]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline rsqrt(float n):
    cdef float d
    _mm_store_ss(&d, _mm_rsqrt_ss(_mm_load_ss(&n)))
    return d

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline corr_coef(float[:, :] arr):
    cdef:
        float[2][256] sample = rand256(arr)
        float[256] x, y, xx, yy, xy
        float corr
        unsigned short i, j, k, l
    with nogil:
        for i in prange(5):
            for j in prange(256):
                if i == 0: x[j] = sample[0][j]
                if i == 1: y[j] = sample[1][j]
                if i == 2: xx[j] = sample[0][j] ** 2
                if i == 3: yy[j] = sample[1][j] ** 2
                if i == 4: xy[j] = sample[0][j] * sample[1][j]
    x[0] = fsum(x)
    y[0] = fsum(y)
    xx[0] = fsum(xx)
    yy[0] = fsum(yy)
    xy[0] = fsum(xy)
    corr = (xy[0] - x[0] * y[0]) * rsqrt((xx[0] - x[0] ** 2) * (yy[0] - y[0] ** 2))
    return corr

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline corr(cnp.ndarray[cnp.float32_t, ndim=2] arr):
    cdef float[:, :] n = arr
    return corr_coef(n)