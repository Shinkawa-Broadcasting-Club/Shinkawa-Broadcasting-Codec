# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False
from cython.parallel import prange

cdef inline unsigned short yuv_709_709_to_rgb_lin_709(unsigned short y, unsigned short u, unsigned short v) nogil:
    
    return