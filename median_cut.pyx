# median_cut.pyx
import numpy as np
cimport numpy as np
cimport cython

#############################
# 補助関数群
#############################

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void swap(np.ndarray[np.int32_t, ndim=1] arr, int i, int j):
	cdef int tmp = arr[i]
	arr[i] = arr[j]
	arr[j] = tmp

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int partition(np.ndarray[np.int32_t, ndim=1] arr, int low, int high,
				   np.ndarray[np.float32_t, ndim=2] pixels, int channel):
	cdef int pivot_index = arr[high]
	cdef int i = low
	cdef int j
	for j in range(low, high):
		if pixels[arr[j], channel] < pixels[pivot_index, channel]:
			swap(arr, i, j)
			i += 1
	swap(arr, i, high)
	return i

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void quicksort_indices(np.ndarray[np.int32_t, ndim=1] arr, int low, int high,
							  np.ndarray[np.float32_t, ndim=2] pixels, int channel):
	if low < high:
		cdef int pi = partition(arr, low, high, pixels, channel)
		quicksort_indices(arr, low, pi - 1, pixels, channel)
		quicksort_indices(arr, pi + 1, high, pixels, channel)

#############################
# 分割判断用の境界計算
#############################
#
# 指定した領域（idx[start:end]）の各チャンネルの最小／最大値を計算するための関数です。
# 戻り値は以下のタプル（14 要素）:
#   (start, end,
#	y_min, y_max,
#	u_min, u_max,
#	v_min, v_max,
#	r_min, r_max,
#	g_min, g_max,
#	b_min, b_max)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple compute_bounds(np.ndarray[np.int32_t, ndim=1] idx, int start, int end, np.ndarray[np.float32_t, ndim=2] pixels):
	cdef int i, j, k, pixel_index
	if start >= end: return (start, end, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	
	pixel_index = idx[start]
	cdef np.ndarray[np.float32_t, ndim=1] minmaxs = np.zeros(12, np.float32_t)
	for i in range(12): minmaxs[i] = pixels[pixel_index, i >> 1]
	
	for i in range(start + 1, end):
		pixel_index = idx[i]
		cdef np.ndarray[np.float32_t, ndim=1] colors = np.zeros(6, np.float32_t)
		for j in range(6): colors[j] = pixels[pixel_index, j]
		for j in range(6):
			k = j >> 1
			if colors[k] < minmaxs[j]: minmaxs[j] = colors[k]
			if colors[k] > minmaxs[j]: minmaxs[j] = colors[k]
		
	return (start, end, minmaxs[0], minmaxs[1], minmaxs[2], minmaxs[3], minmaxs[4], minmaxs[5], minmaxs[6], minmaxs[7], minmaxs[8], minmaxs[9], minmaxs[10], minmaxs[11])

#############################
# メイン関数: median_cut
#############################
#
# この関数は、まず全ピクセルを１つのボックスとし、指定されたボックス数（num_boxes）に
# 達するまで、最もレンジの広いチャンネルに沿って中央値で分割していきます．
#
# その後、各最終ボックスについて各ピクセルの各チャンネル値の平均を代表色として計算し、
# コードブック（代表色のリスト）と、元ピクセルごとにその代表色のインデックス（量子化後のインデックス）
# を返します。
#
# 返り値:
#   (codebook, qindices)
#	 codebook: np.float32 型の (M,6) 配列 (M は実際のボックス数)
#	 qindices: np.int32 型の (num_pixels,) 配列。各要素は対応するピクセルの量子化後のコードブックインデックス
@cython.boundscheck(False)
@cython.wraparound(False)
def median_cut(np.ndarray[np.float32_t, ndim=2] pixels, int num_boxes):
	cdef:
		int num_pixels = pixels.shape[0]
		np.ndarray[np.int32_t, ndim=1] idx = np.empty(num_pixels, dtype=np.int32)
		int i
		list boxes = []
	# 各ピクセルの元インデックスを初期化
	for i in range(num_pixels): idx[i] = i
	# 初期ボックスは全ピクセルを扱う
	boxes.append(compute_bounds(idx, 0, num_pixels, pixels))
	# 指定したボックス数に達するまで分割を繰返す
	while len(boxes) < num_boxes:
		cdef:
			int best_box_index = -1
			float best_range = -1.0
			int best_dim = -1
		for i, box in enumerate(boxes):
			cdef:
				int start, end
				float y_min, y_max, u_min, u_max, v_min, v_max
				float r_min, r_max, g_min, g_max, b_min, b_max
			start, end, y_min, y_max, u_min, u_max, v_min, v_max, r_min, r_max, g_min, g_max, b_min, b_max = box
			cdef:
				float range_y = y_max - y_min
				float range_u = u_max - u_min
				float range_v = v_max - v_min
				float range_r = r_max - r_min
				float range_g = g_max - g_min
				float range_b = b_max - b_min

			if range_y >= range_u and range_y >= range_v and range_y >= range_r and range_y >= range_g and range_y >= range_b:
				curr_dim = 0
				curr_range = range_y
			elif range_u >= range_y and range_u >= range_v and range_u >= range_r and range_u >= range_g and range_u >= range_b:
				curr_dim = 1
				curr_range = range_u
			elif range_v >= range_y and range_v >= range_u and range_v >= range_r and range_v >= range_g and range_v >= range_b:
				curr_dim = 2
				curr_range = range_v
			elif range_r >= range_y and range_r >= range_u and range_r >= range_v and range_r >= range_g and range_r >= range_b:
				curr_dim = 3
				curr_range = range_r
			elif range_g >= range_y and range_g >= range_u and range_g >= range_v and range_g >= range_r and range_g >= range_b:
				curr_dim = 4
				curr_range = range_g
			else:
				curr_dim = 5
				curr_range = range_b

			if curr_range > best_range and (end - start) > 1:
				best_range = curr_range
				best_box_index = i
				best_dim = curr_dim

		if best_box_index == -1:
			break  # これ以上分割できるボックスがなければ終了

		# 分割対象ボックスを取り出し、選んだチャンネルでソート
		box = boxes.pop(best_box_index)
		cdef int start, end
		cdef float dummy1, dummy2, dummy3, dummy4, dummy5, dummy6, dummy7, dummy8, dummy9, dummy10, dummy11, dummy12
		start, end, dummy1, dummy2, dummy3, dummy4, dummy5, dummy6, dummy7, dummy8, dummy9, dummy10, dummy11, dummy12 = box

		quicksort_indices(idx, start, end - 1, pixels, best_dim)
		cdef int median_index = (start + end) >> 1

		boxes.append(compute_bounds(idx, start, median_index, pixels))
		boxes.append(compute_bounds(idx, median_index, end, pixels))
	
	#############################
	# 最終ボックスごとに代表色（各領域内各チャンネルの平均）を計算し，
	# 同時に各ピクセルの量子化後のインデックスを設定します．
	#############################
	cdef np.ndarray[np.int32_t, ndim=1] qindices = np.empty(num_pixels, dtype=np.int32)
	cdef list codebook_list = []  # 一時的に Python のリストに格納

	cdef int b_start, b_end, j, pixel_index, count, box_id
	cdef float sum_y, sum_u, sum_v, sum_r, sum_g, sum_b
	cdef float rep_y, rep_u, rep_v, rep_r, rep_g, rep_b
	cdef float dummy
	# 各ボックスごとに処理
	for box in boxes:
		# compute_bounds の戻り値は (start, end, ... 12個の値)
		b_start, b_end, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy = box
		count = b_end - b_start
		if count <= 0:
			continue

		sum_y = 0.0
		sum_u = 0.0
		sum_v = 0.0
		sum_r = 0.0
		sum_g = 0.0
		sum_b = 0.0
		for j in range(b_start, b_end):
			pixel_index = idx[j]
			sum_y += pixels[pixel_index, 0]
			sum_u += pixels[pixel_index, 1]
			sum_v += pixels[pixel_index, 2]
			sum_r += pixels[pixel_index, 3]
			sum_g += pixels[pixel_index, 4]
			sum_b += pixels[pixel_index, 5]
		rep_y = sum_y / count
		rep_u = sum_u / count
		rep_v = sum_v / count
		rep_r = sum_r / count
		rep_g = sum_g / count
		rep_b = sum_b / count

		# コードブックに代表色を追加
		codebook_list.append( (rep_y, rep_u, rep_v, rep_r, rep_g, rep_b) )
		box_id = len(codebook_list) - 1
		# 該当ボックスに属する各ピクセルに対して量子化後のインデックスを割り当てる
		for j in range(b_start, b_end):
			pixel_index = idx[j]
			qindices[pixel_index] = box_id

	# Python 側で扱いやすいように codebook_list を NumPy 配列に変換
	cdef np.ndarray[np.float32_t, ndim=2] codebook = np.asarray(codebook_list, dtype=np.float32)

	return codebook, qindices
