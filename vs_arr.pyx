import numpy as np
cimport numpy as cnp

cdef inline VStoNP(i):
	cdef:
		cnp.ndarray[cnp.uint16_t, ndim = 4] arr = np.empty((3, i.num_frames, i.height, i.width), np.uint16)
		cnp.ndarray[cnp.uint16_t, ndim = 2] tmp = np.empty((i.height, i.width), np.uint16)
		unsigned short[:, :] t = tmp
		unsigned short[:, :, :] y = arr[0]
		unsigned short[:, :, :] u = arr[1]
		unsigned short[:, :, :] v = arr[2]
		int m
	for m in range(i.num_frames):
		for n in range(3):
			tmp = np.array(i.get_frame(m)[n], copy=False)
			if   n == 0: y[m] = t
			elif n == 1: u[m] = t
			else:        v[m] = t
	return y, u, v

def vs_to_np(i): return VStoNP(i)