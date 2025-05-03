# cython: boundscheck=False, wraparound=False, language_level=3
from sbcfast cimport vs_arr, transform
cimport numpy as cnp
import numpy as np
cpdef tuple vs_to_np(clip): return vs_arr.vs_to_np(clip)
cpdef cnp.ndarray dct_3d_fwd(cnp.ndarray[cnp.int32_t, ndim = 3] arr):
	cdef:
		int x, y, z
		int x = <int> arr.shape[0]
		int y = <int> arr.shape[1]
		int z = <int> arr.shape[2]
		cnp.ndarray[cnp.int32_t, ndim = 3] out = np.empty((x, y, z), np.int32)
		int[:, :, :] inp = arr
		int[:, :, :] tmp = out
	with nogil: transform.dct_3d_fwd(inp, tmp)
	return out