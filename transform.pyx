# cython: boundscheck=False, wraparound=False, nonecheck=False
from cython.parallel import parallel, prange
cdef inline void dct_1d_fwd(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int v15, int v26, int v21, int v28, int v16, int v25, int v22, int v27):
	cdef int v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v17, v18, v19, v20, v23, v24
	v0 = i0 + i7
	v1 = i1 + i6
	v2 = i2 + i5
	v3 = i3 + i4
	v4 = i3 - i4
	v5 = i2 - i5
	v6 = i1 - i6
	v7 = i0 - i7
	
	v8 = v0 + v3
	v9 = v1 + v2
	v10 = v1 - v2
	v11 = v0 - v3
	v12 = v4 + v5
	v13 = c4(v5 + v6)
	v14 = v6 + v7
	
	v15 = v8 + v9
	v16 = v8 - v9
	v17 = c4(v10 + v11)
	v18 = c6(v14 - v12)
	
	v19 = c2_minus_c6(v12) - v18
	v20 = c2_plus_c6(v14) - v18
	v21 = v17 + v11
	v22 = v11 - v17
	v23 = v13 + v7
	v24 = v7 - v13

	v25 = v19 + v24
	v26 = v23 + v20
	v27 = v23 - v20
	v28 = v24 - v19

cdef inline void dct_1d_bwd(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int o0, int o1, int o2, int o3, int o4, int o5, int o6, int o7):
	cdef double C1 = 1.387039845
	cdef double C2 = 1.306562965
	cdef double C3 = 1.175875602
	cdef double C5 = 0.785694958
	cdef double C6 = 0.541196100
	cdef double C7 = 0.275899379

	cdef double x0, x1, x2, x3, x4, x5, x6, x7
	cdef double a0, a1, a2, a3	  # 偶数部中間結果
	cdef double b0, b1, b2, b3	  # 奇数部中間結果
	cdef double t0, t1, t2, t3	  # 共通項：和・差

	# --- EVEN 部分 ---
	x0 = i0
	x1 = i4
	a0 = i0 + i4
	a1 = i0 - i4

	x2 = i2
	x3 = i6

	m1 = x2 * 0.541196100
	m2 = -x3 * 1.306562965
	m3 = (x2 - x3) * 1.847758065

	a2 = m1 - m2
	a3 = m1 + m2 - m3

	# --- ODD 部分 ---
	# (x4,x7) ペア：b0 = x4 * C7 + x7 * C5,  b3 = x4 * C5 - x7 * C7
	m1 = 0.275899379 * i5
	m2 = -0.785694958 * i3
	m3 = 1.061594337 * (i5 - i3)

	b0 = m1 - m2
	b3 = m3 - (m1 + m2)

	# (x5,x6) ペア：b1 = x5 * C1 + x6 * C3,  b2 = x6 * C1 - x5 * C3
	m1 = 1.387039845 * i1
	m2 = -1.175875602 * i7
	m3 = 0.211164243 * (i1 + i7)

	b1 = m1 - m2
	b2 = m3 - (m1 + m2)

	# --- EVEN と ODD の合成 ---

	o0 = a0 + b0
	o7 = a0 - b0
	o1 = a1 + b1
	o6 = a1 - b1
	o2 = a1 + b2
	o5 = a1 - b2
	o3 = a0 + b3
	o4 = a0 - b3

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

cdef inline void dct_3d_fwd(int[:, :, :] arr, int[:, :, :] out, int[8][8] matrix, int q) nogil:
	cdef:
		int i, j, k, l, m, n, a, b, c
		int x = <int> arr.shape[0]
		int y = <int> arr.shape[1]
		int z = <int> arr.shape[2]
		int[8][8] thr
		int thq = (100 - q) << 7
		int v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v17, v18, v19, v20, v23, v24
	if not(0 <= q <= 100): raise ValueError("Quality must be a range [0 - 100]")
	with nogil, parallel():
		for l in prange(8): # threshold
			for m in prange(8):
				thr[l][m] = matrix[l][m] * thq // 100
		for i in prange(x >> 3):
			for j in prange(y >> 3):
				for k in prange(z >> 3):
					for l in prange(8): # DCT(Width-axis)
						for m in prange(8):
							a = i << 3 + l; b = j << 3 + m; c = k << 3
							dct_1d_fwd(arr[a, b, c], arr[a, b, c + 1], arr[a, b, c + 2], arr[a, b, c + 3], arr[a, b, c + 4], arr[a, b, c + 5], arr[a, b, c + 6], arr[a, b, c + 7], out[a, b, c], out[a, b, c + 1], out[a, b, c + 2], out[a, b, c + 3], out[a, b, c + 4], out[a, b, c + 5], out[a, b, c + 6], out[a, b, c + 7])
					for l in prange(8): # DCT(height-axis)
						for m in prange(8):
							a = i << 3 + l; b = j << 3; c = k << 3 + m
							dct_1d_fwd(out[a, b, c], out[a, b + 1, c], out[a, b + 2, c], out[a, b + 3, c], out[a, b + 4, c], out[a, b + 5, c], out[a, b + 6, c], out[a, b + 7, c], out[a, b, c], out[a, b + 1, c], out[a, b + 2, c], out[a, b + 3, c], out[a, b + 4, c], out[a, b + 5, c], out[a, b + 6, c], out[a, b + 7, c])
					for l in prange(8): # Coefficient Filtering
						for m in prange(8):
							for n in prange(8):
								a = i << 3 + l; b = j << 3 + m; c = k << 3 + n
								out[a, b, c] = 0 if out[a, b, c] < thr[m][n] * out[a, j << 3, k << 3] else out[a, b, c]
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

cdef inline void dct_3d_bwd(int[:, :, :] arr, int[:, :, :] out) nogil:
	cdef:
		int i, j, k, l, m, n, a, b, c
		int x = <int> arr.shape[0]
		int y = <int> arr.shape[1]
		int z = <int> arr.shape[2]
		int tmp0, tmp1, tmp2, tmp3, tmp10, tmp11, tmp12, tmp13, t0, t1, t2, t3, z1, z2, z3, z4
		int v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v17, v18, v19
	with nogil, parallel():
		for i in prange(x >> 3):
			for j in prange(y >> 3):
				for k in prange(z >> 3):
					for l in prange(8): # IDCT(Time-axis)
						for m in prange(8):
							a = i << 3; b = j << 3 + l; c = k << 3 + m
							# stage 1
							v0 = arr[a + 0, b, c] + arr[a + 4, b, c]
							v1 = arr[a + 1, b, c] + arr[a + 5, b, c]
							v2 = arr[a + 2, b, c] + arr[a + 6, b, c]
							v3 = arr[a + 3, b, c] + arr[a + 7, b, c]
							v4 = arr[a + 0, b, c] - arr[a + 4, b, c]
							v5 = arr[a + 1, b, c] - arr[a + 5, b, c]
							v6 = arr[a + 2, b, c] - arr[a + 6, b, c]
							v7 = arr[a + 3, b, c] - arr[a + 7, b, c]
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
					for l in prange(8): # IDCT(height-axis)
						for m in prange(8):
							a = i << 3 + l; b = j << 3; c = k << 3 + m
							z1 = c4(out[a, b + 2, c])
							z2 = c4(out[a, b + 6, c])
							z3 = c2_minus_c6(out[a, b + 7, c] - out[a, b + 1, c])
							z4 = c4(out[a, b + 3, c] + out[a, b + 5, c])

							tmp10 = out[a, b, c] + out[a, b + 4, c]
							tmp11 = out[a, b, c] - out[a, b + 4, c]
							tmp12 = z1 - z2
							tmp13 = z1 + z2

							t0 = out[a, b + 1, c] + z3
							t1 = out[a, b + 3, c] - z4
							t2 = out[a, b + 5, c] + z4
							t3 = out[a, b + 7, c] - z3

							tmp0 = tmp10 + tmp13
							tmp1 = tmp11 + tmp12
							tmp2 = tmp11 - tmp12
							tmp3 = tmp10 - tmp13

							out[a, b, c] = (tmp0 + t3) / 8.0
							out[a, b + 7, c] = (tmp0 - t3) / 8.0
							out[a, b + 1, c] = (tmp1 + t2) / 8.0
							out[a, b + 6, c] = (tmp1 - t2) / 8.0
							out[a, b + 2, c] = (tmp2 + t1) / 8.0
							out[a, b + 5, c] = (tmp2 - t1) / 8.0
							out[a, b + 3, c] = (tmp3 + t0) / 8.0
							out[a, b + 4, c] = (tmp3 - t0) >> 3
					for l in prange(8): # IDCT(Width-axis)
						for m in prange(8):
							a = i << 3 + l; b = j << 3 + m; c = k << 3
							z1 = c4(out[a, b, c + 2])
							z2 = c4(out[a, b, c + 6])
							z3 = c2_minus_c6(out[a, b, c + 7] - out[a, b, c + 1])
							z4 = c4(out[a, b, c + 3] + out[a, b, c + 5])

							tmp10 = out[a, b, c] + out[a, b, c + 4]
							tmp11 = out[a, b, c] - out[a, b, c + 4]
							tmp12 = z1 - z2
							tmp13 = z1 + z2

							t0 = out[a, b, c + 1] + z3
							t1 = out[a, b, c + 3] - z4
							t2 = out[a, b, c + 5] + z4
							t3 = out[a, b, c + 7] - z3

							tmp0 = tmp10 + tmp13
							tmp1 = tmp11 + tmp12
							tmp2 = tmp11 - tmp12
							tmp3 = tmp10 - tmp13

							out[a, b, c] = (tmp0 + t3) / 8.0
							out[a, b, c + 7] = (tmp0 - t3) / 8.0
							out[a, b, c + 1] = (tmp1 + t2) / 8.0
							out[a, b, c + 6] = (tmp1 - t2) / 8.0
							out[a, b, c + 2] = (tmp2 + t1) / 8.0
							out[a, b, c + 5] = (tmp2 - t1) / 8.0
							out[a, b, c + 3] = (tmp3 + t0) / 8.0
							out[a, b, c + 4] = (tmp3 - t0) / 8.0


