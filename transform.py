from itertools import product
import numpy as np

def dct_scale():
	scaler = np.zeros((8, 8))
	for i, j in product(range(8), repeat = 2):
		scaler[i, j] = 1 / (np.cos(np.pi / 16 * i) * np.cos(np.pi / 16 * j))
	scaler /= 4
	scaler[0, 0] *= 4
	return scaler

def dct_time(i):
	for j in range(4):
		i[j::8], i[j + 4::8] = i[j::8] + i[j + 4::8], i[j::8] - i[j + 4::8]
	for k in range(2):
		i[k::4], i[k + 2::4] = i[k::4] + i[k + 2::4], i[k::4] - i[k + 2::4]
	i[::2], i[1::2] = i[::2] + i[1::2], i[::2] - i[1::2]
	return i

def dct_fwd(vector):
	v0 = vector[::8] + vector[7::8]
	v1 = vector[1::8] + vector[6::8]
	v2 = vector[2::8] + vector[5::8]
	v3 = vector[3::8] + vector[4::8]
	v4 = vector[3::8] - vector[4::8]
	v5 = vector[2::8] - vector[5::8]
	v6 = vector[1::8] - vector[6::8]
	v7 = vector[::8] - vector[7::8]
	v8 = v0 + v3
	v9 = v1 + v2
	v10 = v1 - v2
	v11 = v0 - v3
	v12 = -v4 - v5
	v13 = (v5 + v6) * 0.7071067811865476
	v14 = v6 + v7
	vector[::8] = v8 + v9
	vector[4::8] = v8 - v9
	v17 = (v10 + v11) * 0.7071067811865476
	v18 = (v12 + v14) * 0.38268343236508984
	v19 = -v12 * 0.5411961001461969 - v18
	v20 = v14 * 1.3065629648763766 - v18
	vector[2::8] = v17 + v11
	vector[6::8] = v11 - v17
	v23 = v13 + v7
	v24 = v7 - v13
	vector[5::8] = v19 + v24
	vector[1::8] = v23 + v20
	vector[7::8] = v23 - v20
	vector[3::8] = v24 - v19
	return vector

def dct_bwd(vector):
	v15 = vector[::8]
	v26 = vector[1::8]
	v21 = vector[2::8]
	v28 = vector[3::8]
	v16 = vector[4::8]
	v25 = vector[5::8]
	v22 = vector[6::8]
	v27 = vector[7::8]
	v19 = (v25 - v28) / 2
	v20 = (v26 - v27) / 2
	v23 = (v26 + v27) / 2
	v24 = (v25 + v28) / 2
	v7  = (v23 + v24) / 2
	v11 = (v21 + v22) / 2
	v13 = (v23 - v24) / 2
	v17 = (v21 - v22) / 2
	v8 = (v15 + v16) / 2
	v9 = (v15 - v16) / 2
	v18 = (v19 - v20) * 0.38268343236508984
	v12 = -(v19 * 1.3065629648763766 - v18)
	v14 = -(v18 - v20 * 0.5411961001461969)
	v6 = v14 - v7
	v5 = v13 * 1.4142135623730951 - v6
	v4 = -v5 - v12
	v10 = v17 * 1.4142135623730951 - v11
	v0 = (v8 + v11) / 2
	v1 = (v9 + v10) / 2
	v2 = (v9 - v10) / 2
	v3 = (v8 - v11) / 2
	vector[::8] = (v0 + v7) / 2
	vector[1::8] = (v1 + v6) / 2
	vector[2::8] = (v2 + v5) / 2
	vector[3::8] = (v3 + v4) / 2
	vector[4::8] = (v3 - v4) / 2
	vector[5::8] = (v2 - v5) / 2
	vector[6::8] = (v1 - v6) / 2
	vector[7::8] = (v0 - v7) / 2
	return vector

def dct_3d_fwd(i):
	i = dct_fwd(dct_fwd(dct_time(i).transpose((2, 0, 1, 3))).transpose((2, 0, 1, 3))).transpose((2, 0, 1, 3))
	for j, k in product(range(8), repeat = 2):
		i[:, j, k] *= dct_scale()[j, k]
	return i

def dct_3d_bwd(i):
	for j, k in product(range(8), repeat = 2):
		i[:, j, k] /= dct_scale()[j, k]
	return dct_bwd(dct_bwd(dct_time(i).transpose((2, 0, 1, 3))).transpose((2, 0, 1, 3))).transpose((2, 0, 1, 3))