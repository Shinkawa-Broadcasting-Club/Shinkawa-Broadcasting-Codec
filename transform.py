from threading import Thread

def add(a, b, c):
	a = b + c

def dct_time(coef):
	v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	for n in range(3):
		if n == 0:
			t1 = Thread(target=add, args=(v0, coef[::8], coef[4::8]))
			t2 = Thread(target=add, args=(v1, coef[1::8], coef[5::8]))
			t3 = Thread(target=add, args=(v2, coef[2::8], coef[6::8]))
			t4 = Thread(target=add, args=(v3, coef[3::8], coef[7::8]))
			t5 = Thread(target=add, args=(v4, coef[::8], -coef[4::8]))
			t6 = Thread(target=add, args=(v5, coef[1::8], -coef[5::8]))
			t7 = Thread(target=add, args=(v6, coef[2::8], -coef[6::8]))
			t8 = Thread(target=add, args=(v7, coef[3::8], -coef[7::8]))
		if n == 1:
			t1 = Thread(target=add, args=(v8, v0, v2))
			t2 = Thread(target=add, args=(v9, v1, v3))
			t3 = Thread(target=add, args=(v10, v0, -v2))
			t4 = Thread(target=add, args=(v11, v1, -v3))
			t5 = Thread(target=add, args=(v12, v4, v6))
			t6 = Thread(target=add, args=(v13, v5, v7))
			t7 = Thread(target=add, args=(v14, v4, -v6))
			t8 = Thread(target=add, args=(v15, v5, -v7))
		if n == 2:
			t1 = Thread(target=add, args=(coef[::8], v8, v9))
			t2 = Thread(target=add, args=(coef[1::8], v8, -v9))
			t3 = Thread(target=add, args=(coef[2::8], v10, v11))
			t4 = Thread(target=add, args=(coef[3::8], v10, -v11))
			t5 = Thread(target=add, args=(coef[4::8], v12, v13))
			t6 = Thread(target=add, args=(coef[5::8], v12, -v13))
			t7 = Thread(target=add, args=(coef[6::8], v14, v15))
			t8 = Thread(target=add, args=(coef[7::8], v14, -v15))
		t1.start()
		t2.start()
		t3.start()
		t4.start()
		t5.start()
		t6.start()
		t7.start()
		t8.start()
		t1.join()
		t2.join()
		t3.join()
		t4.join()
		t5.join()
		t6.join()
		t7.join()
		t8.join()
	del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15
	return coef

def dct_base(coef):
	# stage 1
	v0 = coef[::8] + coef[7::8]
	v1 = coef[1::8] + coef[6::8]
	v2 = coef[2::8] + coef[5::8]
	v3 = coef[3::8] + coef[4::8]
	v4 = coef[3::8] - coef[4::8]
	v5 = coef[2::8] - coef[5::8]
	v6 = coef[1::8] - coef[6::8]
	v7 = coef[::8] - coef[7::8]
	# stage 2
	v8 = v0 + v3
	v9 = v1 + v2
	v10 = v1 - v2
	v11 = v0 - v3
	v12 = -v4 - v5
	v13 = (v5 + v6) * 0.7071067811865476 # √2 / 2
	v14 = v6 + v7
	# stage 3
	v23 = v13 + v7
	v24 = v7 - v13
	v17 = (v10 + v11) * 0.7071067811865476 # √2 / 2
	v18 = (v12 + v14) * 0.3826834323650897 # √(2 - √2) / 2
	# stage 4
	v19 = -v12 * 0.5411961001461969 - v18 # √(2 - √2) * (√2 / 2)
	v20 = v14 * 1.3065629648763766 - v18 # √(2 + √2) * (√2 / 2)
	# stage 5
	coef[::8] = v8 + v9
	coef[1::8] = v23 + v20
	coef[2::8] = v17 + v11
	coef[3::8] = v24 - v19
	coef[4::8] = v8 - v9
	coef[5::8] = v19 + v24
	coef[6::8] = v11 - v17
	coef[7::8] = v23 - v20
	del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v17, v18, v19, v20, v23, v24
	return coef

def dct_3d_fwd(coef):
	for n in range(2):
		if n == 0:
			coef = dct_time(coef)
		else:
			coef = dct_base(coef)
		coef = coef.transpose((2, 0, 1))
	