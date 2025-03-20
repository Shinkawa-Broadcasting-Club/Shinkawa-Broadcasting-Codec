import threading
def add(a, b, c):
	a = b + c

def dct_time(coef):
	v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	for n in range(3):
		if n == 0:
			t1 = threading.Thread(target=add, args=(v0, coef[::8], coef[4::8]))
			t2 = threading.Thread(target=add, args=(v1, coef[1::8], coef[5::8]))
			t3 = threading.Thread(target=add, args=(v2, coef[2::8], coef[6::8]))
			t4 = threading.Thread(target=add, args=(v3, coef[3::8], coef[7::8]))
			t5 = threading.Thread(target=add, args=(v4, coef[::8], -coef[4::8]))
			t6 = threading.Thread(target=add, args=(v5, coef[1::8], -coef[5::8]))
			t7 = threading.Thread(target=add, args=(v6, coef[2::8], -coef[6::8]))
			t8 = threading.Thread(target=add, args=(v7, coef[3::8], -coef[7::8]))
		if n == 1:
			t1 = threading.Thread(target=add, args=(v8, v0, v2))
			t2 = threading.Thread(target=add, args=(v9, v1, v3))
			t3 = threading.Thread(target=add, args=(v10, v0, -v2))
			t4 = threading.Thread(target=add, args=(v11, v1, -v3))
			t5 = threading.Thread(target=add, args=(v12, v4, v6))
			t6 = threading.Thread(target=add, args=(v13, v5, v7))
			t7 = threading.Thread(target=add, args=(v14, v4, -v6))
			t8 = threading.Thread(target=add, args=(v15, v5, -v7))
		if n == 2:
			t1 = threading.Thread(target=add, args=(coef[::8], v8, v9))
			t2 = threading.Thread(target=add, args=(coef[::8], v8, -v9))
			t3 = threading.Thread(target=add, args=(coef[::8], v10, v11))
			t4 = threading.Thread(target=add, args=(coef[::8], v10, -v11))
			t5 = threading.Thread(target=add, args=(coef[::8], v12, v13))
			t6 = threading.Thread(target=add, args=(coef[::8], v12, -v13))
			t7 = threading.Thread(target=add, args=(coef[::8], v14, v15))
			t8 = threading.Thread(target=add, args=(coef[::8], v14, -v15))
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