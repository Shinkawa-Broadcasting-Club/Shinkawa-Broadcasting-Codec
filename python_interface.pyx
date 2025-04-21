# cython: boundscheck=False, wraparound=False, language_level=3
from sbcfast cimport vs_arr, transform
cimport numpy as cnp
import numpy as np
cdef public tuple vs_to_np(clip): return vs_arr.vs_to_np(clip)
cdef public cnp.ndarray dct_3d_fwd(cnp.ndarray[cnp.float32_t, ndim = 3] arr):
	cdef:
		int x, y, z
		int x = <int> arr.shape[0]
		int y = <int> arr.shape[1]
		int z = <int> arr.shape[2]
		cnp.ndarray[cnp.float32_t, ndim = 3] out = np.empty((x, y, z), np.float32)
		float[:, :, :] inp = arr
		float[:, :, :] tmp = out
		float[8][8] mat = [[ 16,  12,  14,  16,  24,  40,  51,  72], 
						   [ 12,  12,  14,  19,  26,  58,  64,  92], 
						   [ 14,  14,  16,  24,  40,  57,  78,  95], 
						   [ 16,  19,  24,  29,  56,  87,  87,  98], 
						   [ 24,  26,  40,  56,  68, 109, 103, 112], 
						   [ 40,  58,  57,  87, 109, 104, 121, 100], 
						   [ 51,  64,  78,  87, 103, 121, 120, 103], 
						   [ 72,  92,  95,  98, 112, 100, 103,  99]]
	with nogil: transform.dct_3d_fwd(inp, tmp, mat, 0)
	return out