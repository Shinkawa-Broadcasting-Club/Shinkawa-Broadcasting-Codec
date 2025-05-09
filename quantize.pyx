# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.stdlib cimport rand, srand
from libc.time cimport time
cdef extern from "omp.h" nogil:
    int omp_get_max_threads()

cpdef inline tuple kmeans_plus_plus_quantization(cnp.ndarray[cnp.int32_t, ndim=1] data, int k=32, int max_iter=0x7FFFFFFF):
    cdef int n = data.shape[0]
    cdef int i, j, iter
    cdef np.ndarray[cnp.int32_t, ndim=1] cents = np.empty(k, dtype=np.int32)
    cdef np.ndarray[cnp.int64_t, ndim=1] dist = np.empty(n, dtype=np.int64)
    cdef np.ndarray[cnp.int64_t, ndim=1] cumulative = np.empty(n, dtype=np.int64)
    cdef np.ndarray[cnp.int64_t, ndim=1] sum_vals = np.zeros(k, dtype=np.int64)
    cdef np.ndarray[cnp.int32_t, ndim=1] counts = np.zeros(k, dtype=np.int32)
    cdef np.ndarray[cnp.int32_t, ndim=1] labels = np.empty(n, dtype=np.int32)
    cdef np.ndarray[cnp.int32_t, ndim=1] quantized = np.empty(n, dtype=np.int32)
    cdef int[:] data_view = data
    cdef int[:] labels_view = labels
    cdef int[:] q = quantized
    cdef int[:] centroids = cents
    cdef long[:] distances = dist
    cdef int diff, tmp, best
    cdef int best_label
    cdef long total, r
    cdef int changed, prev_centroid
    cdef int candidate
    srand(<unsigned int>time(NULL))
    i = rand() % n
    centroids[0] = data_view[i]
    for i in range(n):
        diff = data_view[i] - centroids[0]
        distances[i] = diff * diff
    for j in range(1, k):
        cumulative[0] = distances[0]
        for i in range(1, n): cumulative[i] = cumulative[i-1] + distances[i]
        total = cumulative[n-1]
        if total == 0: candidate = data_view[rand() % n]
        else:
            r = rand() % total
            i = 0
            while i < n and cumulative[i] < r: i += 1
            if i == n: i = n - 1
            candidate = data_view[i]
        centroids[j] = candidate
        for i in range(n):
            diff = data_view[i] - candidate
            tmp = diff * diff
            if tmp < distances[i]:
                distances[i] = tmp
    iter = 0
    while iter < max_iter:
        with nogil:
            for i in prange(n, schedule='static'):
                best_label = 0
                diff = data_view[i] - centroids[0]
                best = diff * diff
                for j in range(1, k):
                    diff = data_view[i] - centroids[j]
                    tmp = diff * diff
                    if tmp < best:
                        best = tmp
                        best_label = j
                labels_view[i] = best_label
        for j in range(k):
            sum_vals[j] = 0
            counts[j] = 0
        for i in range(n):
            j = labels_view[i]
            sum_vals[j] += data_view[i]
            counts[j] += 1
        changed = 0
        for j in range(k):
            if counts[j] > 0:
                prev_centroid = centroids[j]
                # 完全整数計算のため、切り捨ての結果になります
                centroids[j] = <int>(sum_vals[j] // counts[j])
                if centroids[j] != prev_centroid:
                    changed += 1
        if changed == 0:
            break
        iter += 1

    # ----------------------------
    # 最終量子化：各点を最も近い（更新後）中心に置き換える
    # ----------------------------
    
    with nogil:
        for i in prange(n, schedule='static'):
            best_label = 0
            diff = data_view[i] - centroids[0]
            best = diff * diff
            for j in range(1, k):
                diff = data_view[i] - centroids[j]
                tmp = diff * diff
                if tmp < best:
                    best = tmp
                    best_label = j
            q[i] = centroids[best_label]
    
    return centroids, quantized
