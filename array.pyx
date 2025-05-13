# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from cython cimport sizeof

cdef inline int*** alloc_3d_int_arr(int x, int y, int z) nogil:
    cdef int*** arr = <int***> malloc(x * sizeof(int**))
    if arr == NULL: return
    cdef int i, j, k, l
    for i in prange(x):
        arr[i] = <int**> malloc(y * sizeof(int*))
        if arr[i] == NULL:
            for j in prange(i): free(arr[j])
            return
        for j in prange(y):
            arr[i][j] = <int*> malloc(z * sizeof(int))
            if arr[i][j] == NULL:
                for k in prange(j): free(arr[i][k])
                for k in prange(i):
                    for l in prange(y):
                        if arr[k][l] != NULL: free(arr[k][l])
                    free(arr[k])
                free(arr)
                return
    return arr

cdef inline unsigned short*** alloc_3d_ushort_arr(int x, int y, int z) nogil:
    cdef unsigned short*** arr = <unsigned short***> malloc(x * sizeof(unsigned short**))
    if arr == NULL: return
    cdef int i, j, k, l
    for i in prange(x):
        arr[i] = <unsigned short**> malloc(y * sizeof(unsigned short*))
        if arr[i] == NULL:
            for j in prange(i): free(arr[j])
            return
        for j in prange(y):
            arr[i][j] = <unsigned short*> malloc(z * sizeof(unsigned short))
            if arr[i][j] == NULL:
                for k in prange(j): free(arr[i][k])
                for k in prange(i):
                    for l in prange(y):
                        if arr[k][l] != NULL: free(arr[k][l])
                    free(arr[k])
                free(arr)
                return
    return arr

cdef inline unsigned char*** alloc_3d_ushort_arr(int x, int y, int z) nogil:
    cdef unsigned char*** arr = <unsigned char***> malloc(x * sizeof(unsigned char**))
    if arr == NULL: return
    cdef int i, j, k, l
    for i in prange(x):
        arr[i] = <unsigned char**> malloc(y * sizeof(unsigned char*))
        if arr[i] == NULL:
            for j in prange(i): free(arr[j])
            return
        for j in prange(y):
            arr[i][j] = <unsigned char*> malloc(z * sizeof(unsigned char))
            if arr[i][j] == NULL:
                for k in prange(j): free(arr[i][k])
                for k in prange(i):
                    for l in prange(y):
                        if arr[k][l] != NULL: free(arr[k][l])
                    free(arr[k])
                free(arr)
                return
    return arr

cdef inline void free_3d_int_arr(int*** arr) nogil:
    cdef int i, j, k, l
    free(arr)
    if arr != NULL: return

cdef inline void free_3d_ushort_arr(unsigned short*** arr) nogil:
    cdef int i, j, k, l
    free(arr)
    if arr != NULL: return

cdef inline void free_3d_ushort_arr(unsigned char*** arr) nogil:
    cdef int i, j, k, l
    free(arr)
    if arr != NULL: return
