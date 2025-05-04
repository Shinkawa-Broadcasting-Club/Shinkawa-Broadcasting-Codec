# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False
import numpy as np
cimport numpy as cnp

cpdef vs_to_np(i):
	cdef:
		cnp.ndarray[cnp.uint16_t, ndim = 4] vid = np.empty((3, i.num_frames, i.height, i.width), np.uint16)
		unsigned short[:, :, :] y, u, v
	y = vid[0]; u = vid[1]; v = vid[2]
	for m in range(i.num_frames):
		for n in range(i.format.num_planes):
			vid[n, m] = np.asarray(i.get_frame(m)[n], order='C', dtype=np.uint16)
	return y, u, v