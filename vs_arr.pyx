import numpy as np
cimport numpy as cnp

cdef get_vsframe(int m, int n, i): return np.array(i.get_frame(m)[n], copy=False)
cdef vs_to_np(i):
	cdef:
		cnp.ndarray[cnp.uint16_t, ndim = 3] y = np.empty((i.num_frames, i.height, i.width))
		cnp.ndarray[cnp.uint16_t, ndim = 3] u = np.empty((i.num_frames, i.height, i.width))
		cnp.ndarray[cnp.uint16_t, ndim = 3] v = np.empty((i.num_frames, i.height, i.width))
		cnp.ndarray[cnp.uint16_t, ndim = 2] t = np.empty((i.height, i.width))
	for m in range(i.num_frames):
		for n in range(i.format.num_planes):
			t = get_vsframe(m, n, i)
			if n == 0: y[m] = t
			if n == 1: u[m] = t
			if n == 2: v[m] = t
	return y, u, v
