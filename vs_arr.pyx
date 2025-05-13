# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False
import numpy as np
cimport numpy as cnp
from sbcsimd.array cimport alloc_3d_ushort_arr

cpdef vs_to_np(i):
	cdef:
		int m
		int a = <int> i.num_frames
		int b = <int> i.height
		int c = <int> i.width
		int d = <int> i.format.subsampling_h
		int e = <int> i.format.subsampling_w
		cnp.ndarray[cnp.uint16_t, ndim = 3] yt = np.empty((a, b, c), np.uint16)
		cnp.ndarray[cnp.uint16_t, ndim = 3] ut = np.empty((a, d, e), np.uint16)
		cnp.ndarray[cnp.uint16_t, ndim = 3] vt = np.empty((a, d, e), np.uint16)
		unsigned short*** y, u, v
	with nogil:
		y = alloc_3d_ushort_arr(a, b, c)
		u = alloc_3d_ushort_arr(a, d, e)
		v = alloc_3d_ushort_arr(a, d, e)
	for m in range(i.num_frames):
		yt[m] = np.asarray(i.get_frame(m)[0], order='C', dtype=np.uint16)
		ut[m] = np.asarray(i.get_frame(m)[1], order='C', dtype=np.uint16)
		vt[m] = np.asarray(i.get_frame(m)[2], order='C', dtype=np.uint16)
	return y, u, v