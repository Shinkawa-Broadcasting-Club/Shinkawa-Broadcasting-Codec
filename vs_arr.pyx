# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False
import numpy as np
cimport numpy as cnp

cpdef vs_to_np(i):
	cdef:
		cnp.ndarray[cnp.uint16_t, ndim = 3] y = np.empty((i.num_frames, i.height, i.width), np.uint16)
		cnp.ndarray[cnp.uint16_t, ndim = 3] u = np.empty((i.num_frames, i.height, i.width), np.uint16)
		cnp.ndarray[cnp.uint16_t, ndim = 3] v = np.empty((i.num_frames, i.height, i.width), np.uint16)
		int m
	for m in range(i.num_frames):
		y[m] = np.asarray(i.get_frame(m)[0], order='C', dtype=np.uint16)
		u[m] = np.asarray(i.get_frame(m)[1], order='C', dtype=np.uint16)
		v[m] = np.asarray(i.get_frame(m)[2], order='C', dtype=np.uint16)
	return y, u, v