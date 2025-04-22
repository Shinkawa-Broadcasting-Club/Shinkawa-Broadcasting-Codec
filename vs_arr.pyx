import numpy as np
cimport numpy as cnp

cdef inline VStoNP(i):
	cdef:
		cnp.ndarray[cnp.uint16_t, ndim = 4] arr = np.empty((3, i.num_frames, i.height, i.width), np.uint16)
		cnp.ndarray[cnp.uint16_t, ndim = 2] t = np.empty((i.height, i.width), np.uint16)
		unsigned short[:, :, :] y = arr[0]
		unsigned short[:, :, :] u = arr[1]
		unsigned short[:, :, :] v = arr[2]
		int m
	for m in range(i.num_frames):
		y[m] = np.array(i.get_frame(m)[0], copy=False)
		u[m] = np.array(i.get_frame(m)[1], copy=False)
		v[m] = np.array(i.get_frame(m)[2], copy=False)
	return y, u, v

def vs_to_np(i): return VStoNP(i)