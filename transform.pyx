# cython: boundscheck=False, wraparound=False, nonecheck=False
from cython.parallel import parallel, prange

cdef inline int sft8_sgn(int n) nogil:
	if n > 0x7FFFFFFF: return -(-n >> 8)
	else: return n >> 8

cdef inline void dct_1d_fwd(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int o0, int o1, int o2, int o3, int o4, int o5, int o6, int o7) nogil:
	cdef int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, z1, z2, z3, z4, z5, z11, z13
	tmp0 = i0 + i7
	tmp7 = i0 - i7
	tmp1 = i1 + i6
	tmp6 = i1 - i6
	tmp2 = i2 + i5
	tmp5 = i2 - i5
	tmp3 = i3 + i4
	tmp4 = i3 - i4

	tmp10 = tmp0 + tmp3
	tmp11 = tmp1 + tmp2
	tmp12 = tmp1 - tmp2
	tmp13 = tmp0 - tmp3
	tmp14 = tmp4 + tmp5
	tmp15 = tmp5 + tmp6
	tmp16 = tmp6 + tmp7
	
	z1 = sft8_sgn((tmp12 + tmp13) * 181)
	z3 = sft8_sgn(tmp15 * 181)
	z5 = (tmp14 - tmp16) * 98

	z2 = sft8_sgn(tmp14 * 139 + z5)
	z4 = sft8_sgn(tmp16 * 334 + z5)
	z11 = tmp7 + z3
	z13 = tmp7 - z3

	o0 = tmp10 + tmp11
	o1 = z11 + z4
	o2 = tmp13 + z1
	o3 = z13 - z2
	o4 = tmp10 - tmp11
	o6 = tmp13 - z1
	o5 = z13 + z2
	o7 = z11 - z4

cdef inline void dct_1d_bwd(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int o0, int o1, int o2, int o3, int o4, int o5, int o6, int o7) nogil:
	cdef int x0, x1, x2, x3, x4, x5, x6, x7, a0, a1, a2, a3, a4, a5, a6, a7, a10, a11, a12, a13
	# Step 1: Load input
	x0 = i0
	x1 = i1
	x2 = i2
	x3 = i3
	x4 = i4
	x5 = i5
	x6 = i6
	x7 = i7
	
	# Step 2: Process even coefficients (x0, x2, x4, x6)
	a0 = i0 + i4
	a1 = i0 - i4
	a2 = i2 + i6
	a3 = sft8_sgn(i2 - i6 * 181)  # 1st multiplication
	
	x0 = a0 + a2
	x2 = a0 - a2
	x4 = a1 + a3
	x6 = a1 - a3
	
	# Step 3: Process odd coefficients (x1, x3, x5, x7)
	a10 = i1 + i7
	a11 = i5 + i3
	a12 = i5 - i3
	a13 = i1 - i7
	
	a4 = a10 + a11
	a5 = a10 - a11
	a6 = a13 + a12
	a7 = a13 - a12

	x1 = sft8_sgn(a4 * 362)
	x3 = sft8_sgn(a6 * 98)
	x5 = sft8_sgn(a5 * 139)
	x7 = sft8_sgn(a7 * 334)
	
	# Step 4: Final butterfly
	o0 = x0 + x1
	o1 = x4 + x3
	o2 = x2 + x5
	o3 = x6 + x7
	o4 = x6 - x7
	o5 = x2 - x5
	o6 = x4 - x3
	o7 = x0 - x1

cdef inline void dct_3d_fwd(int[:, :, :] arr, int[:, :, :] out) nogil:
	cdef:
		int i, j, k, l, m, n, a, b, c
		int x = <int> arr.shape[0]
		int y = <int> arr.shape[1]
		int z = <int> arr.shape[2]
		int[8][8] thr
	with nogil, parallel():
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
					for l in prange(8): # DCT(Time-axis)
						for m in prange(8):
							a = i << 3; b = j << 3 + l; c = k << 3 + m
							# stage 1
							out[a + 0, b, c] = out[a + 0, b, c] + out[a + 4, b, c]
							out[a + 1, b, c] = out[a + 1, b, c] + out[a + 5, b, c]
							out[a + 2, b, c] = out[a + 2, b, c] + out[a + 6, b, c]
							out[a + 3, b, c] = out[a + 3, b, c] + out[a + 7, b, c]
							out[a + 4, b, c] = out[a + 0, b, c] - out[a + 4, b, c]
							out[a + 5, b, c] = out[a + 1, b, c] - out[a + 5, b, c]
							out[a + 6, b, c] = out[a + 2, b, c] - out[a + 6, b, c]
							out[a + 7, b, c] = out[a + 3, b, c] - out[a + 7, b, c]
							# stage 2
							out[a + 0, b, c] = out[a + 0, b, c] + out[a + 2, b, c]
							out[a + 1, b, c] = out[a + 1, b, c] + out[a + 3, b, c]
							out[a + 2, b, c] = out[a + 0, b, c] - out[a + 2, b, c]
							out[a + 3, b, c] = out[a + 1, b, c] - out[a + 3, b, c]
							out[a + 4, b, c] = out[a + 4, b, c] + out[a + 6, b, c]
							out[a + 5, b, c] = out[a + 5, b, c] + out[a + 7, b, c]
							out[a + 6, b, c] = out[a + 4, b, c] - out[a + 6, b, c]
							out[a + 7, b, c] = out[a + 5, b, c] - out[a + 7, b, c]
							# stage 3
							out[a + 0, b, c] = out[a + 0, b, c] + out[a + 1, b, c]
							out[a + 1, b, c] = out[a + 0, b, c] - out[a + 1, b, c]
							out[a + 2, b, c] = out[a + 2, b, c] + out[a + 3, b, c]
							out[a + 3, b, c] = out[a + 2, b, c] - out[a + 3, b, c]
							out[a + 4, b, c] = out[a + 4, b, c] + out[a + 5, b, c]
							out[a + 5, b, c] = out[a + 4, b, c] - out[a + 5, b, c]
							out[a + 6, b, c] = out[a + 6, b, c] + out[a + 7, b, c]
							out[a + 7, b, c] = out[a + 6, b, c] - out[a + 7, b, c]

cdef inline void dct_3d_bwd(int[:, :, :] arr, int[:, :, :] out) nogil:
	cdef:
		int i, j, k, l, m, n, a, b, c
		int x = <int> arr.shape[0]
		int y = <int> arr.shape[1]
		int z = <int> arr.shape[2]
	with nogil, parallel():
		for i in prange(x >> 3):
			for j in prange(y >> 3):
				for k in prange(z >> 3):
					for l in prange(8): # IDCT(Time-axis)
						for m in prange(8):
							a = i << 3; b = j << 3 + l; c = k << 3 + m
							# stage 1
							out[a + 0, b, c] = arr[a + 0, b, c] + arr[a + 4, b, c]
							out[a + 1, b, c] = arr[a + 1, b, c] + arr[a + 5, b, c]
							out[a + 2, b, c] = arr[a + 2, b, c] + arr[a + 6, b, c]
							out[a + 3, b, c] = arr[a + 3, b, c] + arr[a + 7, b, c]
							out[a + 4, b, c] = arr[a + 0, b, c] - arr[a + 4, b, c]
							out[a + 5, b, c] = arr[a + 1, b, c] - arr[a + 5, b, c]
							out[a + 6, b, c] = arr[a + 2, b, c] - arr[a + 6, b, c]
							out[a + 7, b, c] = arr[a + 3, b, c] - arr[a + 7, b, c]
							# stage 2
							out[a + 0, b, c] = out[a + 0, b, c] + out[a + 2, b, c]
							out[a + 1, b, c] = out[a + 1, b, c] + out[a + 3, b, c]
							out[a + 2, b, c] = out[a + 0, b, c] - out[a + 2, b, c]
							out[a + 3, b, c] = out[a + 1, b, c] - out[a + 3, b, c]
							out[a + 4, b, c] = out[a + 4, b, c] + out[a + 6, b, c]
							out[a + 5, b, c] = out[a + 5, b, c] + out[a + 7, b, c]
							out[a + 6, b, c] = out[a + 4, b, c] - out[a + 6, b, c]
							out[a + 7, b, c] = out[a + 5, b, c] - out[a + 7, b, c]
							# stage 3
							out[a + 0, b, c] = out[a + 0, b, c] + out[a + 1, b, c]
							out[a + 1, b, c] = out[a + 0, b, c] - out[a + 1, b, c]
							out[a + 2, b, c] = out[a + 2, b, c] + out[a + 3, b, c]
							out[a + 3, b, c] = out[a + 2, b, c] - out[a + 3, b, c]
							out[a + 4, b, c] = out[a + 4, b, c] + out[a + 5, b, c]
							out[a + 5, b, c] = out[a + 4, b, c] - out[a + 5, b, c]
							out[a + 6, b, c] = out[a + 6, b, c] + out[a + 7, b, c]
							out[a + 7, b, c] = out[a + 6, b, c] - out[a + 7, b, c]
					for l in prange(8): # IDCT(height-axis)
						for m in prange(8):
							a = i << 3 + l; b = j << 3; c = k << 3 + m
							dct_1d_bwd(out[a, b, c], out[a, b + 1, c], out[a, b + 2, c], out[a, b + 3, c], out[a, b + 4, c], out[a, b + 5, c], out[a, b + 6, c], out[a, b + 7, c], out[a, b, c], out[a, b + 1, c], out[a, b + 2, c], out[a, b + 3, c], out[a, b + 4, c], out[a, b + 5, c], out[a, b + 6, c], out[a, b + 7, c])
					for l in prange(8): # IDCT(Width-axis)
						for m in prange(8):
							a = i << 3 + l; b = j << 3 + m; c = k << 3
							dct_1d_bwd(out[a, b, c], out[a, b, c + 1], out[a, b, c + 2], out[a, b, c + 3], out[a, b, c + 4], out[a, b, c + 5], out[a, b, c + 6], out[a, b, c + 7], out[a, b, c], out[a, b, c + 1], out[a, b, c + 2], out[a, b, c + 3], out[a, b, c + 4], out[a, b, c + 5], out[a, b, c + 6], out[a, b, c + 7])


