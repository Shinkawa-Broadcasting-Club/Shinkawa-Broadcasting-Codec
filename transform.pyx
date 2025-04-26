# cython: boundscheck=False, wraparound=False, nonecheck=False
from cython.parallel import parallel, prange

cdef inline int c2_minus_c6(int i):
	cdef int n
	n = ~(i - 1) if i > 0x7FFFFFFF else i
	n = n >> 1 + n >> 5 + n >> 7 + n >> 9 + n >> 14 + n >> 15 + n >> 17 + n >> 20 + n >> 21
	n = ~n + 1 if i > 0x7FFFFFFF else n
	return n

cdef inline int c2_plus_c6(int i):
	cdef int n
	n = ~(i - 1) if i > 0x7FFFFFFF else i
	n += n >> 2 + n >> 5 + n >> 6 + n >> 7 + n >> 12 + n >> 14 + n >> 17 + n >> 18 + n >> 19
	n = ~n + 1 if i > 0x7FFFFFFF else n
	return n

cdef inline int c4(int i):
	cdef int n
	n = ~(i - 1) if i > 0x7FFFFFFF else i
	n = n >> 1 + n >> 3 + n >> 4 + n >> 6 + n >> 8 + n >> 9 + n >> 11 + n >> 12 + n >> 14 + n >> 16 + n >> 17 + n >> 19 + n >> 20
	n = ~n + 1 if i > 0x7FFFFFFF else n
	return n

cdef inline int c6(int i):
	cdef int n
	n = ~(i - 1) if i > 0x7FFFFFFF else i
	n = n >> 2 + n >> 3 + n >> 8 + n >> 10 + n >> 13 + n >> 14 + n >> 15 + n >> 16 + n >> 18 + n >> 20
	n = ~n + 1 if i > 0x7FFFFFFF else n
	return n

cdef inline void dct_3d_fwd(int[:, :, :] arr, int[:, :, :] out, int[8][8] matrix, int q):
	cdef:
		int i, j, k, l, m, n, a, b, c
		int x = <int> arr.shape[0]
		int y = <int> arr.shape[1]
		int z = <int> arr.shape[2]
		float[8][8] thr
		float[8][8] mul = [[1.        , 0.25489779, 0.27059805, 0.30067244, 0.35355339, 0.44998811, 0.65328148, 1.28145772],
						   [0.25489779, 0.06497288, 0.06897484, 0.07664074, 0.09011998, 0.11470097, 0.16652001, 0.32664074],
						   [0.27059805, 0.06897484, 0.0732233 , 0.08136138, 0.09567086, 0.12176591, 0.1767767 , 0.34675996],
						   [0.30067244, 0.07664074, 0.08136138, 0.09040392, 0.10630376, 0.13529903, 0.19642374, 0.38529903],
						   [0.35355339, 0.09011998, 0.09567086, 0.10630376, 0.125     , 0.15909482, 0.23096988, 0.45306372],
						   [0.44998811, 0.11470097, 0.12176591, 0.13529903, 0.15909482, 0.2024893 , 0.2939689 , 0.57664074],
						   [0.65328148, 0.16652001, 0.1767767 , 0.19642374, 0.23096988, 0.2939689 , 0.4267767 , 0.8371526 ],
						   [1.28145772, 0.32664074, 0.34675996, 0.38529903, 0.45306372, 0.57664074, 0.8371526 , 1.6421339 ]]
		int v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v17, v18, v19, v20, v23, v24
	if not(0 <= q <= 100): raise ValueError("Quality must be a range [0 - 100]")
	with nogil, parallel():
		for l in prange(8): # threshold
			for m in prange(8):
				thr[l][m] = matrix[l][m] * (1 - q * 0.01) / (mul[l][m] * 2048)
		for i in prange(x >> 3):
			for j in prange(y >> 3):
				for k in prange(z >> 3):
					for l in prange(8): # DCT(Width-axis)
						for m in prange(8):
							a = i << 3 + l; b = j << 3 + m; c = k << 3
							# stage 1
							v0 = arr[a, b, c + 0] + arr[a, b, c + 7]
							v1 = arr[a, b, c + 1] + arr[a, b, c + 6]
							v2 = arr[a, b, c + 2] + arr[a, b, c + 5]
							v3 = arr[a, b, c + 3] + arr[a, b, c + 4]
							v4 = arr[a, b, c + 3] - arr[a, b, c + 4]
							v5 = arr[a, b, c + 2] - arr[a, b, c + 5]
							v6 = arr[a, b, c + 1] - arr[a, b, c + 6]
							v7 = arr[a, b, c + 0] - arr[a, b, c + 7]
							# stage 2
							v8 = v0 + v3
							v9 = v1 + v2
							v10 = v1 - v2
							v11 = v0 - v3
							v12 = v4 + v5
							v13 = c4(v5 + v6)
							v14 = v6 + v7
							# stage 3
							v17 = c4(v10 + v11)
							v18 = c6(v14 - v12)
							# stage 4
							v19 = c2_minus_c6(v12) - v18
							v20 = c2_plus_c6(v14) - v18
							# stage 5
							v23 = v13 + v7
							v24 = v7 - v13
							# stage 6
							out[a, b, c + 0] = v8 + v9
							out[a, b, c + 1] = v23 + v20
							out[a, b, c + 2] = v17 + v11
							out[a, b, c + 3] = v24 - v19
							out[a, b, c + 4] = v8 - v9
							out[a, b, c + 5] = v19 + v24
							out[a, b, c + 6] = v11 - v17
							out[a, b, c + 7] = v23 - v20
					for l in prange(8): # DCT(height-axis)
						for m in prange(8):
							a = i << 3 + l; b = j << 3; c = k << 3 + m
							# stage 1
							v0 = out[a, b + 0, c] + out[a, b + 7, c]
							v1 = out[a, b + 1, c] + out[a, b + 6, c]
							v2 = out[a, b + 2, c] + out[a, b + 5, c]
							v3 = out[a, b + 3, c] + out[a, b + 4, c]
							v4 = out[a, b + 3, c] - out[a, b + 4, c]
							v5 = out[a, b + 2, c] - out[a, b + 5, c]
							v6 = out[a, b + 1, c] - out[a, b + 6, c]
							v7 = out[a, b + 0, c] - out[a, b + 7, c]
							# stage 2
							v8 = v0 + v3
							v9 = v1 + v2
							v10 = v1 - v2
							v11 = v0 - v3
							v12 = v4 + v5
							v13 = c4(v5 + v6)
							v14 = v6 + v7
							# stage 3
							v17 = c4(v10 + v11)
							v18 = c6(v14 - v12)
							# stage 4
							v19 = c2_minus_c6(v12) - v18
							v20 = c2_plus_c6(v14) - v18
							# stage 5
							v23 = v13 + v7
							v24 = v7 - v13
							# stage 6
							out[a, b + 0, c] = v8 + v9
							out[a, b + 1, c] = v23 + v20
							out[a, b + 2, c] = v17 + v11
							out[a, b + 3, c] = v24 - v19
							out[a, b + 4, c] = v8 - v9
							out[a, b + 5, c] = v19 + v24
							out[a, b + 6, c] = v11 - v17
							out[a, b + 7, c] = v23 - v20
					for l in prange(8): # Coefficient Filtering
						for m in prange(8):
							for n in prange(8):
								a = i << 3 + l; b = j << 3 + m; c = k << 3 + n
								out[a, b, c] = 0 if out[a, b, c] < thr[m][n] * out[a, j << 3, k << 3] else out[a, b, c] * mul[m][n]
					for l in prange(8): # DCT(Time-axis)
						for m in prange(8):
							a = i << 3; b = j << 3 + l; c = k << 3 + m
							# stage 1
							v0 = out[a + 0, b, c] + out[a + 4, b, c]
							v1 = out[a + 1, b, c] + out[a + 5, b, c]
							v2 = out[a + 2, b, c] + out[a + 6, b, c]
							v3 = out[a + 3, b, c] + out[a + 7, b, c]
							v4 = out[a + 0, b, c] - out[a + 4, b, c]
							v5 = out[a + 1, b, c] - out[a + 5, b, c]
							v6 = out[a + 2, b, c] - out[a + 6, b, c]
							v7 = out[a + 3, b, c] - out[a + 7, b, c]
							# stage 2
							v10 = v0 + v2
							v11 = v1 + v3
							v12 = v0 - v2
							v13 = v1 - v3
							v14 = v4 + v6
							v17 = v5 + v7
							v18 = v4 - v6
							v19 = v5 - v7
							# stage 3
							out[a + 0, b, c] = v10 + v11
							out[a + 1, b, c] = v10 - v11
							out[a + 2, b, c] = v12 + v13
							out[a + 3, b, c] = v12 - v13
							out[a + 4, b, c] = v14 + v17
							out[a + 5, b, c] = v14 - v17
							out[a + 6, b, c] = v18 + v19
							out[a + 7, b, c] = v18 - v19