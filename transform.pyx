# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False
from cython.parallel import prange, parallel
cimport numpy as cnp
import numpy as np

cdef inline int sft8_sgn(int n) nogil:
    if n > 0x7FFFFFFF: return -(-n >> 8)
    else: return n >> 8

cdef inline void dct_1d_fwd(int[512] i, int step, int offset) nogil:
    cdef int[8] l, t
    cdef int[7] u
    cdef int[3] v
    cdef int[4] w
    cdef int m
    for m in prange(8): l[m] = m * step + offset

    t[0] = i[l[0]] + i[l[7]]
    t[1] = i[l[1]] + i[l[6]]
    t[2] = i[l[2]] + i[l[5]]
    t[3] = i[l[3]] + i[l[4]]
    t[4] = i[l[3]] - i[l[4]]
    t[5] = i[l[2]] - i[l[5]]
    t[6] = i[l[1]] - i[l[6]]
    t[7] = i[l[0]] - i[l[7]]

    u[0] = t[0] + t[3]
    u[1] = t[1] + t[2]
    u[2] = t[1] - t[2]
    u[3] = t[0] - t[3]
    u[4] = t[4] + t[5]
    u[5] = t[5] + t[6]
    u[6] = t[6] + t[7]

    v[0] = sft8_sgn((u[2] + u[3]) * 181)
    v[1] = sft8_sgn(u[5] * 181)
    v[2] = (u[4] - u[6]) * 98

    w[0] = sft8_sgn(u[4] * 139 + v[2])
    w[1] = sft8_sgn(u[6] * 334 + v[2])
    w[2] = t[7] + v[1]
    w[3] = t[7] - v[1]

    i[l[0]] = u[0] + u[1]
    i[l[1]] = w[2] + w[1]
    i[l[2]] = u[3] + v[0]
    i[l[3]] = w[3] - w[0]
    i[l[4]] = u[0] - u[1]
    i[l[5]] = w[3] + w[0]
    i[l[6]] = u[3] - v[0]
    i[l[7]] = w[2] - w[1]

cdef inline void dct_1d_bwd(int[512] i, int step, int offset) nogil:
    cdef int[8] x, a, l
    cdef int m
    for m in prange(8): l[m] = m * step + offset

    a[0] = i[l[0]] + i[l[4]]
    a[1] = i[l[0]] - i[l[4]]
    a[2] = i[l[2]] + i[l[6]]
    a[3] = sft8_sgn((i[l[2]] - i[l[6]]) * 181)
    a[4] = i[l[1]] + i[l[7]]
    a[5] = i[l[5]] + i[l[3]]
    a[6] = i[l[5]] - i[l[3]]
    a[7] = i[l[1]] - i[l[7]]

    x[0] = a[0] + a[2]
    x[2] = a[0] - a[2]
    x[4] = a[1] + a[3]
    x[6] = a[1] - a[3]
    x[1] = sft8_sgn((a[4] + a[5]) * 362)
    x[3] = sft8_sgn((a[7] + a[6]) * 98)
    x[5] = sft8_sgn((a[4] - a[5]) * 139)
    x[7] = sft8_sgn((a[7] - a[6]) * 334)

    i[l[0]] = x[0] + x[1]
    i[l[1]] = x[4] + x[3]
    i[l[2]] = x[2] + x[5]
    i[l[3]] = x[6] + x[7]
    i[l[4]] = x[6] - x[7]
    i[l[5]] = x[2] - x[5]
    i[l[6]] = x[4] - x[3]
    i[l[7]] = x[0] - x[1]

cdef inline void dct_time(int[512] i, int step, int offset) nogil:
    cdef int[8] j, k, l
    cdef int m
    for m in prange(8): l[m] = m * step + offset

    j[0] = i[l[0]] + i[l[4]]
    j[1] = i[l[1]] + i[l[5]]
    j[2] = i[l[2]] + i[l[6]]
    j[3] = i[l[3]] + i[l[7]]
    j[4] = i[l[0]] - i[l[4]]
    j[5] = i[l[1]] - i[l[5]]
    j[6] = i[l[2]] - i[l[6]]
    j[7] = i[l[3]] - i[l[7]]

    k[0] = j[0] + j[2]
    k[1] = j[1] + j[3]
    k[2] = j[4] + j[6]
    k[3] = j[5] + j[7]
    k[4] = j[0] - j[2]
    k[5] = j[1] - j[3]
    k[6] = j[4] - j[6]
    k[7] = j[5] - j[7]

    i[l[0]] = k[0] + k[1]
    i[l[1]] = k[0] - k[1]
    i[l[2]] = k[2] + k[3]
    i[l[3]] = k[2] - k[3]
    i[l[4]] = k[4] + k[5]
    i[l[5]] = k[4] - k[5]
    i[l[6]] = k[6] + k[7]
    i[l[7]] = k[6] - k[7]

cdef inline void vrdif_3dfwd(int[512] arr) nogil:
    cdef int i
    with parallel():
        for i in range(64): dct_1d_fwd(arr, 1, i << 3)
        for i in range(64): dct_1d_fwd(arr, 8, ((i >> 3) << 6) + (i & 7))
        for i in range(64): dct_time(arr, 64, i)

cdef inline void vrdif_3dbwd(int[512] arr) nogil:
    cdef int i
    with parallel():
        for i in range(64): dct_1d_bwd(arr, 1, i << 3)
        for i in range(64): dct_1d_bwd(arr, 8, ((i >> 3) << 6) + (i & 7))
        for i in range(64): dct_time(arr, 64, i)

cpdef inline int[:, :, :] dct_3d_fwd(unsigned short[:, :, :] arr):
    cdef int x = <int> arr.shape[0]
    cdef int y = <int> arr.shape[1]
    cdef int z = <int> arr.shape[2]
    cdef cnp.ndarray[cnp.int32_t, ndim=3] o = np.empty((x, y, z), np.int32)
    cdef int[:, :, :] out = o
    cdef int[512] t
    cdef int i, j, k, l, m, n
    with nogil, parallel():
        for l in prange(x >> 3):
            for m in prange(y >> 3):
                for n in prange(z >> 3):
                    for i in prange(8):
                        for j in prange(8):
                            for k in prange(8):
                                t[(i << 6) + (j << 3) + k] = arr[(l << 3) + i, (m << 3) + j, (n << 3) + k]
                    vrdif_3dfwd(t)
                    for i in prange(8):
                        for j in prange(8):
                            for k in prange(8):
                                out[(l << 3) + i, (m << 3) + j, (n << 3) + k] = t[(i << 6) + (j << 3) + k]
    return out

cpdef inline int[:, :, :] dct_3d_bwd(unsigned short[:, :, :] arr):
    cdef int x = <int> arr.shape[0]
    cdef int y = <int> arr.shape[1]
    cdef int z = <int> arr.shape[2]
    cdef cnp.ndarray[cnp.int32_t, ndim=3] o = np.empty((x, y, z), np.int32)
    cdef int[:, :, :] out = o
    cdef int[512] t
    cdef int i, j, k, l, m, n
    with nogil, parallel():
        for l in prange(x >> 3):
            for m in prange(y >> 3):
                for n in prange(z >> 3):
                    for i in prange(8):
                        for j in prange(8):
                            for k in prange(8):
                                t[(i << 6) + (j << 3) + k] = arr[(l << 3) + i, (m << 3) + j, (n << 3) + k]
                    vrdif_3dbwd(t)
                    for i in prange(8):
                        for j in prange(8):
                            for k in prange(8):
                                out[(l << 3) + i, (m << 3) + j, (n << 3) + k] = t[(i << 6) + (j << 3) + k]
    return out
