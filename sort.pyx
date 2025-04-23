# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as cnp
from cython.parallel import prange
ctypedef float float32_t
cnp.import_array()
cdef inline void swap(float32_t* arr, int i, int j):
    cdef float32_t temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp

cdef inline int partition(float32_t* arr, int low, int high):
    cdef:
        float32_t pivot = arr[high]
        int i = (low - 1), j
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            swap(arr, i, j)
    swap(arr, i + 1, high)
    return (i + 1)

cdef void quicksort_recursive(float32_t* arr, int low, int high):
    if low < high:
        cdef int pi = partition(arr, low, high)
        quicksort_recursive(arr, low, pi - 1)
        quicksort_recursive(arr, pi + 1, high)

def quicksort_flat(cnp.ndarray[cnp.float32_t, ndim=3] arr):
    cdef int n = <int>arr.size
    cdef float32_t* arr_ptr = <float32_t*>arr.data
    quicksort_recursive(arr_ptr, 0, n - 1)