# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False
cpdef inline int sft8_sgn(int n) nogil:
	if n > 0x7FFFFFFF: return -(-n >> 8)
	else: return n >> 8

cpdef inline int[8] dct_1d_fwd(int[8] i) nogil:
	cdef int[22] t
	cdef int[8] o
	t[0] = i[0] + i[7]
	t[1] = i[1] + i[6]
	t[2] = i[2] + i[5]
	t[3] = i[3] + i[4]
	t[4] = i[3] - i[4]
	t[5] = i[2] - i[5]
	t[6] = i[1] - i[6]
	t[7] = i[0] - i[7]
	t[8] = t[0] + t[3]
	t[9] = t[1] + t[2]
	t[10] = t[1] - t[2]
	t[11] = t[0] - t[3]
	t[12] = t[4] + t[5]
	t[13] = t[5] + t[6]
	t[14] = t[6] + t[7]
	t[15] = sft8_sgn((t[10] + t[11]) * 181)
	t[16] = sft8_sgn(t[13] * 181)
	t[17] = (t[12] - t[14]) * 98
	t[18] = sft8_sgn(t[12] * 139 + t[17])
	t[19] = sft8_sgn(t[14] * 334 + t[17])
	t[20] = t[7] + t[16]
	t[21] = t[7] - t[16]
	o[0] = t[8] + t[9]
	o[1] = t[20] + t[19]
	o[2] = t[11] + t[15]
	o[3] = t[21] - t[18]
	o[4] = t[8] - t[9]
	o[5] = t[21] + t[18]
	o[6] = t[11] - t[15]
	o[7] = t[20] - t[19]
	return o

cpdef inline int[8] dct_1d_bwd(int[8] i) nogil:
	cdef int[8] o
	cdef int[16] t
	t[0] = i[0] + i[4]
	t[1] = i[0] - i[4]
	t[2] = i[2] + i[6]
	t[3] = sft8_sgn((i[2] - i[6]) * 181)
	t[4] = i[1] + i[7]
	t[5] = i[5] + i[3]
	t[6] = i[5] - i[3]
	t[7] = i[1] - i[7]
	t[8] = t[0] + t[2]
	t[9] = sft8_sgn((t[4] + t[5]) * 362)
	t[10] = t[0] - t[2]
	t[11] = sft8_sgn((t[7] + t[6]) * 98)
	t[12] = t[1] + t[3]
	t[13] = sft8_sgn((t[4] - t[5]) * 139)
	t[14] = t[1] - t[3]
	t[15] = sft8_sgn((t[7] - t[6]) * 334)
	o[0] = t[8] + t[9]
	o[1] = t[12] + t[11]
	o[2] = t[10] + t[13]
	o[3] = t[14] + t[15]
	o[4] = t[14] - t[15]
	o[5] = t[10] - t[13]
	o[6] = t[12] - t[11]
	o[7] = t[8] - t[9]
	return o
