# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange, parallel

cdef inline int calc_range(int[:] arr):
    cdef int samples = <int> arr.shape[1]

cdef inline int mean_cut(int[:, :] arr, int q):
    cdef int n = <int> arr.shape[0]
    cdef int samples = <int> arr.shape[1]
