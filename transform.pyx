# cython: boundscheck=False, wraparound=False, nonecheck=False
from cython.parallel import parallel, prange

cdef inline int c2_minus_c6(int i) nogil:
	cdef int n
	n = -i if i > 0x7FFFFFFF else i
	n = (n >> 1) + (n >> 5) + (n >> 7) + (n >> 9) + (n >> 12) - (n >> 14) - (n >> 18) + (n >> 20)
	n = -n if i > 0x7FFFFFFF else n
	return n

cdef inline int c2_plus_c6(int i) nogil:
	cdef int n
	n = -i if i > 0x7FFFFFFF else i
	n += (n >> 2) + (n >> 4) - (n >> 7) + (n >> 9) - (n >> 14) - (n >> 16) - (n >> 19) + (n >> 21)
	n = -n if i > 0x7FFFFFFF else n
	return n

cdef inline int c4(int i) nogil:
	cdef int n
	n = -i if i > 0x7FFFFFFF else i
	n += (n >> 6) + (n >> 8) + (n >> 14) + (n >> 16) - (n >> 2) - (n >> 4) - (n >> 20)
	n = -n if i > 0x7FFFFFFF else n
	return n

cdef inline int c6(int i) nogil:
	cdef int n
	n = -i if i > 0x7FFFFFFF else i
	n = (n >> 1) - (n >> 3) + (n >> 7) - (n >> 13) - (n >> 17) + (n >> 21)
	n = -n if i > 0x7FFFFFFF else n
	return n

cdef inline int mul_dct(int l, int m, int i) nogil:
	cdef int j, o
	cdef int[23] p, m
	cdef int n = -i if i > 0x7FFFFFFF else i
	for j in prange(23): p[j] = 0x7FFFFFFF; m[j] = 0x7FFFFFFF
	# shaft
	if l == 0 and m == 0: return n
	if l == 1 and m == 1: p = [4, 9, 11, 15, 20]
	if l == 2 and m == 2: p = [4, 6]; m = [8, 10, 16, 18]
	if l == 3 and m == 3: p = [3, 11, 14, 16]; m = [5, 8, 18, 21]
	if l == 4 and m == 4: return n >> 3
	if l == 5 and m == 5: p = []; m = []
	if l == 6 and m == 6:
		return
	if l == 7 and m == 7:
		return
	# 0-line
	if (l == 0 and m == 1) or (l == 1 and m == 0):
		return
	if (l == 0 and m == 2) or (l == 2 and m == 0):
		return
	if (l == 0 and m == 3) or (l == 3 and m == 0):
		return
	if (l == 0 and m == 4) or (l == 4 and m == 0):
		return
	if (l == 0 and m == 5) or (l == 5 and m == 0):
		return
	if (l == 0 and m == 6) or (l == 6 and m == 0):
		return
	if (l == 0 and m == 7) or (l == 7 and m == 0):
		return
	# 1-line
	if (l == 1 and m == 2) or (l == 2 and m == 1):
		return
	if (l == 1 and m == 3) or (l == 3 and m == 1):
		return
	if (l == 1 and m == 4) or (l == 4 and m == 1):
		return
	if (l == 1 and m == 5) or (l == 5 and m == 1):
		return
	if (l == 1 and m == 6) or (l == 6 and m == 1):
		return
	if (l == 1 and m == 7) or (l == 7 and m == 1):
		return
	# 2-line
	if (l == 2 and m == 3) or (l == 3 and m == 2):
		return
	if (l == 2 and m == 4) or (l == 4 and m == 2):
		return
	if (l == 2 and m == 5) or (l == 5 and m == 2):
		return
	if (l == 2 and m == 6) or (l == 6 and m == 2):
		return
	if (l == 2 and m == 7) or (l == 7 and m == 2):
		return
	# 3-line
	if (l == 3 and m == 4) or (l == 4 and m == 3):
		return
	if (l == 3 and m == 5) or (l == 5 and m == 3):
		return
	if (l == 3 and m == 6) or (l == 6 and m == 3):
		return
	if (l == 3 and m == 7) or (l == 7 and m == 3):
		return
	# 4-line
	if (l == 4 and m == 5) or (l == 5 and m == 4):
		return
	if (l == 4 and m == 6) or (l == 6 and m == 4):
		return
	if (l == 4 and m == 7) or (l == 7 and m == 4):
		return
	# 5-line
	if (l == 5 and m == 6) or (l == 6 and m == 5):
		return
	if (l == 5 and m == 7) or (l == 7 and m == 5):
		return
	# 6-line
	if (l == 6 and m == 7) or (l == 7 and m == 6):
		return
	for j in range(23):
		if p[j] != 0x7FFFFFFF: o += n >> p[j]
	for j in range(23):
		if m[j] != 0x7FFFFFFF: o -= n >> m[j]
	if i > 0x7FFFFFFF: return -o
	else: return o

cdef inline int mul_dct_rep(int l, int m, int i) nogil:
	return

cdef inline void dct_3d_fwd(int[:, :, :] arr, int[:, :, :] out, int[8][8] matrix, int q) nogil:
	cdef:
		int i, j, k, l, m, n, a, b, c
		int x = <int> arr.shape[0]
		int y = <int> arr.shape[1]
		int z = <int> arr.shape[2]
		float[8][8] thr
		int v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v17, v18, v19, v20, v23, v24
	if not(0 <= q <= 100): raise ValueError("Quality must be a range [0 - 100]")
	with nogil, parallel():
		for l in prange(8): # threshold
			for m in prange(8):
				thr[l][m] = mul_dct_rep(l, m, matrix[l][m] * (1 - q * 0.01) / 2048)
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
								out[a, b, c] = 0 if out[a, b, c] < thr[m][n] * out[a, j << 3, k << 3] else mul_dct(m, n, out[a, b, c])
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

cdef inline void idct_3d_fwd(int[:, :, :] arr, int[:, :, :] out):
	cdef:
		int i, j, k, l, m, n, a, b, c
		int x = <int> arr.shape[0]
		int y = <int> arr.shape[1]
		int z = <int> arr.shape[2]
		int a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3
		int v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v17, v18, v19
	with nogil, parallel():
		for i in prange(x >> 3):
			for j in prange(y >> 3):
				for k in prange(z >> 3):
					for l in prange(8): # IDCT(Width-axis)
						for m in prange(8):
							a = i << 3 + l; b = j << 3 + m; c = k << 3
							# stage 1
							a0 = arr[a, b, c + 0] + arr[a, b, c + 4]
							a1 = arr[a, b, c + 0] - arr[a, b, c + 4]
							t = (arr[a, b, c + 2] + arr[a, b, c + 6]) * 0.541196100
							b0 = arr[a, b, c + 7] + arr[a, b, c + 1]
							b1 = arr[a, b, c + 5] + arr[a, b, c + 3]
							b2 = arr[a, b, c + 7] - arr[a, b, c + 1]
							b3 = arr[a, b, c + 5] - arr[a, b, c + 3]
							# stage 2
							a2 = t - (arr[a, b, c + 6] * 1.847759065)
							a3 = t + (arr[a, b, c + 2] * 0.765366865)
							# stage 3
							c0 = a0 + a3
							c3 = a0 - a3
							c1 = a1 + a2
							c2 = a1 - a2
							d0 = b0 + b1
							d1 = 0.382683433 * (b2 - b3)
							d2 = 0.382683433 * (b2 + b3)
							d3 = b0 - b1
							# stage 4
							arr[a, b, c + 0] = c0 + d0
							arr[a, b, c + 1] = c1 + d1
							arr[a, b, c + 2] = c2 + d2
							arr[a, b, c + 3] = c3 + d3
							arr[a, b, c + 4] = c3 - d3
							arr[a, b, c + 5] = c2 - d2
							arr[a, b, c + 6] = c1 - d1
							arr[a, b, c + 7] = c0 - d0
					for l in prange(8): # IDCT(height-axis)
						for m in prange(8):
							a = i << 3 + l; b = j << 3; c = k << 3 + m
							# stage 1
							a0 = arr[a, b + 0, c] + arr[a, b + 4, c]
							a1 = arr[a, b + 0, c] - arr[a, b + 4, c]
							t = (arr[a, b + 2, c] + arr[a, b + 6, c]) * 0.541196100
							b0 = arr[a, b + 7, c] + arr[a, b + 1, c]
							b1 = arr[a, b + 5, c] + arr[a, b + 3, c]
							b2 = arr[a, b + 7, c] - arr[a, b + 1, c]
							b3 = arr[a, b + 5, c] - arr[a, b + 3, c]
							# stage 2
							a2 = t - (arr[a, b, c + 6] * 1.847759065)
							a3 = t + (arr[a, b, c + 2] * 0.765366865)
							# stage 3
							c0 = a0 + a3
							c3 = a0 - a3
							c1 = a1 + a2
							c2 = a1 - a2
							d0 = b0 + b1
							d1 = 0.382683433 * (b2 - b3)
							d2 = 0.382683433 * (b2 + b3)
							d3 = b0 - b1
							# stage 4
							arr[a, b + 0, c] = c0 + d0
							arr[a, b + 1, c] = c1 + d1
							arr[a, b + 2, c] = c2 + d2
							arr[a, b + 3, c] = c3 + d3
							arr[a, b + 4, c] = c3 - d3
							arr[a, b + 5, c] = c2 - d2
							arr[a, b + 6, c] = c1 - d1
							arr[a, b + 7, c] = c0 - d0
					for l in prange(8): # IDCT(Time-axis)
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