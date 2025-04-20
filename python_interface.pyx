# cython: boundscheck=False, wraparound=False, language_level=3
from sbcfast cimport vs_arr, transform
cimport numpy as cnp
cdef public tuple vs_to_np(clip): return vs_arr.vs_to_np(clip)
cdef public float[:, :, :] dct_3d_fwd(cnp.ndarray[cnp.float32_t, ndim = 3] arr):
	cdef:
		cnp.ndarray[cnp.ndarray] out
		float[:, :, :] inp = arr