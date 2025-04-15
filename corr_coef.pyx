# cython: boundscheck=False, wraparound=False, nonecheck=False
cimport cython
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport rand, srand, malloc, free
from libc.string cimport memcpy
from libc.time cimport time
from cython.parallel import parallel, prange
cdef extern from "nmmintrin.h" nogil:
    ctypedef struct __m128: pass
    __m128 _mm_load_ss (const float* mem_addr)
    __m128 _mm_loadu_ps (const float* mem_addr)
    __m128 _mm_rsqrt_ss (__m128 a)
    __m128 _mm_add_ps (__m128 a, __m128 b)
    __m128 _mm_hadd_ps (__m128 a, __m128 b)
    void _mm_store_ss (float* mem_addr, __m128 a)
    void _mm_storeu_ps (float* mem_addr, __m128 a)

cdef inline mul256_flt(float f) nogil:
    cdef:
        int d
        float n
    memcpy(&d, &f, sizeof(f))
    d += 0x04000000
    memcpy(&n, &d, sizeof(d))
    return n

cdef inline rand256(float[:, :] arr):
    cdef:
        int N = arr.shape[1]
        float[2][256] out
        int* indices = <int*> malloc(N * sizeof(int))
        int i, j, k, tmp
    if N < 256: raise ValueError("入力配列の第2軸（列数）は256以上である必要があります。")
    if indices == NULL: raise MemoryError("インデックス配列のメモリ確保に失敗しました。")
    with nogil, parallel():
        for i in prange(N): indices[i] = i
    srand(<unsigned int> time(NULL))
    for i in range(256):
        j = i + rand() % (N - i)
        tmp = indices[i]
        indices[i] = indices[j]
        indices[j] = tmp
        for k in range(2): out[k][i] = arr[k, indices[i]]
    free(indices)
    return out

cdef inline fsum(float[256] arr):
    cdef:
        __m128[64] tmp
        float[4] dst
        int i
    with nogil, parallel():
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

cdef inline rsqrt(float n):
    cdef float d
    _mm_store_ss(&d, _mm_rsqrt_ss(_mm_load_ss(&n)))
    return d

cdef inline corr_coef(float[:, :] arr):
    cdef:
        float[2][256] sample = rand256(arr)
        float[5][256] tmp
        int x, y, xx, yy, xy
        unsigned short i, j, k, l
    with nogil, parallel():
        for i in prange(5):
            for j in prange(256):
                if i == 0: tmp[0][j] = sample[0][j]
                if i == 1: tmp[1][j] = sample[1][j]
                if i == 2: tmp[2][j] = sample[0][j] ** 2
                if i == 3: tmp[3][j] = sample[1][j] ** 2
                if i == 4: tmp[4][j] = sample[0][j] * sample[1][j]
    x = fsum(tmp[0])
    y = fsum(tmp[1])
    xx = mul256_flt(fsum(tmp[2]))
    yy = mul256_flt(fsum(tmp[3]))
    xy = mul256_flt(fsum(tmp[4]))
    return (xy - x * y) * rsqrt((xx - x ** 2) * (yy - y ** 2))

cpdef inline corr(cnp.ndarray[cnp.float32_t, ndim=2] arr):
    cdef float[:, :] n = arr
    return corr_coef(n)