# cython: boundscheck=False, wraparound=False, nonecheck=False
from cython.parallel import parallel, prange

cdef inline void dct_fwd(float[8] vector, float[8] out) nogil:
	cdef:
		float c4 = 0.70710677
		float c6 = 0.38268343
		float a2 = 0.5411961
		float a4 = 1.306563
		# stage 1
		float v0 = vector[0] + vector[7]
		float v1 = vector[1] + vector[6]
		float v2 = vector[2] + vector[5]
		float v3 = vector[3] + vector[4]
		float v4 = vector[3] - vector[4]
		float v5 = vector[2] - vector[5]
		float v6 = vector[1] - vector[6]
		float v7 = vector[0] - vector[7]
		# stage 2
		float v8 = v0 + v3
		float v9 = v1 + v2
		float v10 = v1 - v2
		float v11 = v0 - v3
		float v12 = v4 + v5
		float v13 = (v5 + v6) * c4
		float v14 = v6 + v7
		# stage 3
		float v17 = (v10 + v11) * c4
		float v18 = (v14 - v12) * c6
		# stage 4
		float v19 = v12 * a2 - v18
		float v20 = v14 * a4 - v18
		# stage 5
		float v23 = v13 + v7
		float v24 = v7 - v13
	out[0] = v8 + v9
	out[1] = v23 + v20
	out[2] = v17 + v11
	out[3] = v24 - v19
	out[4] = v8 - v9
	out[5] = v19 + v24
	out[6] = v11 - v17
	out[7] = v23 - v20

cdef inline void dct_time_fwd(float[8] inp, float[8] out) nogil:
	cdef int i
	with nogil, parallel():
		for i in prange(8): out[i] = inp[i] + inp[i + 4] if i < 4 else inp[i - 4] - inp[i]
		for i in prange(8): out[i] = out[i] + out[i + 2] if i & 3 < 2 else out[i - 2] - out[i]
		for i in prange(8): out[i] = out[i] + out[i + 1] if i & 1 == 0 else out[i - 1] - out[i]

cdef inline void dct3dfwd(float[8][8][8] arr, float[8][8][8] out, float[8][8] matrix) nogil:
	cdef:
		int i, j, k
		float[8][8][8] tmp
	with nogil, parallel():
		for i in prange(8):
			for j in prange(8):
				dct_fwd(arr[i][j], tmp[i][j])
		for i in prange(8):
			for j in prange(8):
				for k in prange(8):
					tmp[i][j][k] = tmp[k][i][j] # (0, 1, 2) -> (1, 2, 0)
		for i in prange(8):
			for j in prange(8):
				dct_fwd(tmp[i][j], tmp[i][j])
		for i in prange(8):
			for j in prange(8):
				for k in prange(8):
					tmp[i][j][k] = tmp[j][k][i] # (1, 2, 0) -> (0, 1, 2)
		for i in prange(8):
			for j in prange(8):
				for k in prange(8):
					tmp[i][j][k] = 0 if tmp[i][j][k] < matrix[j][k]	else tmp[i][j][k]
		for i in prange(8):
			for j in prange(8):
				for k in prange(8):
					tmp[i][j][k] = tmp[j][k][i] # (0, 1, 2) -> (2, 0, 1)
		for i in prange(8):
			for j in prange(8):
				dct_time_fwd(tmp[i][j], tmp[i][j])
		for i in prange(8):
			for j in prange(8):
				for k in prange(8):
					out[i][j][k] = tmp[k][i][j] # (2, 0, 1) -> (0, 1, 2)

cdef inline void dct_3d_fwd(float[:, :, :] arr, float[:, :, :] out, float[:, :] matrix, float q):
	cdef:
		int i, j, k, l, m, n
		int x = <int> arr.shape[0]
		int y = <int> arr.shape[1]
		int z = <int> arr.shape[2]
		float[8][8][8] dct
		float[8][8] mat
	if not(0 <= q <= 100): raise ValueError("Quality must be a range [0 - 100]")
	with nogil, parallel():
		for l in prange(8):
			for m in prange(8):
				mat[l][m] = matrix[l, m] - matrix[l, m] * q * 0.01
		for i in prange(x >> 3):
			for j in prange(y >> 3):
				for k in prange(z >> 3):
					for l in prange(8):
						for m in prange(8):
							for n in prange(8):
								dct[l][m][n] = arr[i << 3 + l, j << 3 + m, k << 3 + n]
					dct3dfwd(dct, dct, mat)
					for l in prange(8):
						for m in prange(8):
							for n in prange(8):
								out[i << 3 + l, j << 3 + m, k << 3 + n] = dct[l][m][n]

