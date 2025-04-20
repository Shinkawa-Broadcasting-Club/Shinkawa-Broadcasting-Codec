# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrtf, fabsf
import numpy as np
cimport numpy as np

cdef inline void compute_box_bounds(int* order, float* pts, int start, int end, float* min_x, float* max_x, float* min_y, float* max_y) nogil:
	cdef int k, idx_local
	if start >= end:
		min_x[0] = 0.0
		max_x[0] = 0.0
		min_y[0] = 0.0
		max_y[0] = 0.0
		return
	idx_local = order[start]
	min_x[0] = pts[idx_local*2]
	max_x[0] = pts[idx_local*2]
	min_y[0] = pts[idx_local*2+1]
	max_y[0] = pts[idx_local*2+1]
	for k in range(start+1, end):
		idx_local = order[k]
		if pts[idx_local*2] < min_x[0]: min_x[0] = pts[idx_local*2]
		elif pts[idx_local*2] > max_x[0]: max_x[0] = pts[idx_local*2]
		if pts[idx_local*2+1] < min_y[0]: min_y[0] = pts[idx_local*2+1]
		elif pts[idx_local*2+1] > max_y[0]: max_y[0] = pts[idx_local*2+1]

cdef inline int partition(int* order, float* pts, int left, int right, int dim) nogil:
	cdef:
		int i = left, j, idx_local, tmp
		float pivot_val = pts[ order[right-1] * 2 + dim ]
	for j in range(left, right-1):
		idx_local = order[j]
		if pts[idx_local*2 + dim] < pivot_val:
			tmp = order[i]
			order[i] = order[j]
			order[j] = tmp
			i += 1
	tmp = order[i]
	order[i] = order[right-1]
	order[right-1] = tmp
	return i

cdef inline int quickselect(int* order, float* pts, int left, int right, int k, int dim) nogil:
	cdef int pivot_index
	while right - left > 1:
		pivot_index = partition(order, pts, left, right, dim)
		if pivot_index == k: return pivot_index
		elif pivot_index < k: left = pivot_index + 1
		else: right = pivot_index
	return left

cdef inline void compute_centroid(int* order, float* pts, int start, int end, float* centroid_x, float* centroid_y) nogil:
	cdef:
		float sumx = 0.0, sumy = 0.0
		int k, idx_local, n = end - start
	if n == 0:
		centroid_x[0] = 0.0
		centroid_y[0] = 0.0
		return
	for k in range(start, end):
		idx_local = order[k]
		sumx += pts[idx_local * 2]
		sumy += pts[idx_local * 2 + 1]
	centroid_x[0] = sumx / n
	centroid_y[0] = sumy / n

cdef inline void median_cut_clustering_internal(float* pts, int num_pts, int n_clusters_target, int* order, int* cluster_assign_temp, float* codebook_temp, int* actual_n_clusters) nogil:
	cdef:
		int max_boxes = n_clusters_target
		int* box_start = <int*> malloc(max_boxes * sizeof(int))
		int* box_end   = <int*> malloc(max_boxes * sizeof(int))
		float* box_min_x = <float*> malloc(max_boxes * sizeof(float))
		float* box_max_x = <float*> malloc(max_boxes * sizeof(float))
		float* box_min_y = <float*> malloc(max_boxes * sizeof(float))
		float* box_max_y = <float*> malloc(max_boxes * sizeof(float))
	if not box_start or not box_end or not box_min_x or not box_max_x or not box_min_y or not box_max_y:
		actual_n_clusters[0] = 0
		return
	cdef:
		int num_boxes = 1
		int i, selected_box, mid, dim, n_in_box
		float range_x, range_y, current_range, max_range, centroid_x_val, centroid_y_val
	box_start[0] = 0
	box_end[0] = num_pts
	compute_box_bounds(order, pts, 0, num_pts, &box_min_x[0], &box_max_x[0], &box_min_y[0], &box_max_y[0])
	cdef:
		int current_box_start, current_box_end
		float current_box_min_x, current_box_max_x, current_box_min_y, current_box_max_y
	while num_boxes < n_clusters_target:
		max_range = -1.0
		selected_box = -1
		for i in range(num_boxes):
			n_in_box = box_end[i] - box_start[i]
			if n_in_box <= 1: continue
			range_x = box_max_x[i] - box_min_x[i]
			range_y = box_max_y[i] - box_min_y[i]
			current_range = range_x if range_x >= range_y else range_y
			if current_range > max_range:
				max_range = current_range
				selected_box = i
		if selected_box == -1: break
		current_box_start = box_start[selected_box]
		current_box_end = box_end[selected_box]
		current_box_min_x = box_min_x[selected_box]
		current_box_max_x = box_max_x[selected_box]
		current_box_min_y = box_min_y[selected_box]
		current_box_max_y = box_max_y[selected_box]
		if selected_box != num_boxes - 1:
			box_start[selected_box] = box_start[num_boxes - 1]
			box_end[selected_box] = box_end[num_boxes - 1]
			box_min_x[selected_box] = box_min_x[num_boxes - 1]
			box_max_x[selected_box] = box_max_x[num_boxes - 1]
			box_min_y[selected_box] = box_min_y[num_boxes - 1]
			box_max_y[selected_box] = box_max_y[num_boxes - 1]
		range_x = current_box_max_x - current_box_min_x
		range_y = current_box_max_y - current_box_min_y
		if range_x >= range_y: dim = 0
		else: dim = 1
		mid = (current_box_start + current_box_end) >> 1
		quickselect(order, pts, current_box_start, current_box_end, mid, dim)
		box_start[num_boxes - 1] = current_box_start
		box_end[num_boxes - 1] = mid
		compute_box_bounds(order, pts, box_start[num_boxes - 1], box_end[num_boxes - 1], &box_min_x[num_boxes - 1], &box_max_x[num_boxes - 1], &box_min_y[num_boxes - 1], &box_max_y[num_boxes - 1])
		box_start[num_boxes] = mid
		box_end[num_boxes] = current_box_end
		compute_box_bounds(order, pts, box_start[num_boxes], box_end[num_boxes], &box_min_x[num_boxes], &box_max_x[num_boxes], &box_min_y[num_boxes], &box_max_y[num_boxes])
		num_boxes += 1
	actual_n_clusters[0] = num_boxes
	for i in range(num_boxes):
		compute_centroid(order, pts, box_start[i], box_end[i], &centroid_x_val, &centroid_y_val)
		codebook_temp[i*2] = centroid_x_val
		codebook_temp[i*2+1] = centroid_y_val
		for j in range(box_start[i], box_end[i]): cluster_assign_temp[order[j]] = i
	free(box_start)
	free(box_end)
	free(box_min_x)
	free(box_max_x)
	free(box_min_y)
	free(box_max_y)

cdef inline float euclidean_distance_pts(float* p1, float* p2) nogil: return sqrtf((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
cpdef quantize_data(float[:, :, :] data, int max_clusters_to_test):
	cdef int H = data.shape[0]
	cdef int W = data.shape[1]
	cdef int total = H * W
	cdef int num_non_zero_points = 0
	cdef int i, j, k

	if max_clusters_to_test < 1: raise ValueError("max_clusters_to_testは1以上の整数である必要があります。")
	for i in range(H):
		for j in range(W):
			if not (data[i,j,0] == 0.0 and data[i,j,1] == 0.0): num_non_zero_points += 1
	if num_non_zero_points == 0:
		print("Info: 全ての点が (0, 0) です。")
		out_indices = np.zeros((H, W, 1), dtype=np.intc)
		out_codebook = np.empty((0,2), dtype=np.float32)
		return out_codebook, out_indices
	cdef float* points = <float*> malloc(num_non_zero_points * 2 * sizeof(float))
	cdef int* orig_i = <int*> malloc(num_non_zero_points * sizeof(int))
	cdef int* orig_j = <int*> malloc(num_non_zero_points * sizeof(int))
	cdef int* order = <int*> malloc(num_non_zero_points * sizeof(int))

	if not points or not orig_i or not orig_j or not order:
		if points: free(points)
		if orig_i: free(orig_i)
		if orig_j: free(orig_j)
		if order: free(order)
		raise MemoryError("メモリ確保に失敗しました。")

	cdef int idx = 0
	for i in range(H):
		for j in range(W):
			if data[i,j,0] == 0.0 and data[i,j,1] == 0.0:
				continue
			points[idx*2]   = data[i,j,0]
			points[idx*2+1] = data[i,j,1]
			orig_i[idx] = i
			orig_j[idx] = j
			order[idx] = idx
			idx += 1
	cdef list mae_values = []
	cdef int n_clusters_k
	cdef float total_mae
	cdef float point_mae
	cdef int actual_n_clusters_k
	cdef int assigned_cluster_idx
	cdef int* cluster_assign_temp = <int*> malloc(num_non_zero_points * sizeof(int))
	cdef float* codebook_temp = <float*> malloc(max_clusters_to_test * 2 * sizeof(float))
	if not cluster_assign_temp or not codebook_temp:
		if points: free(points)
		if orig_i: free(orig_i)
		if orig_j: free(orig_j)
		if order: free(order)
		if cluster_assign_temp: free(cluster_assign_temp)
		if codebook_temp: free(codebook_temp)
		raise MemoryError("一時配列のメモリ確保に失敗しました。")
	cdef float centroid_x, centroid_y, average_mae
	for n_clusters_k in range(1, max_clusters_to_test + 1):
		for i in range(num_non_zero_points): order[i] = i
		median_cut_clustering_internal(points, num_non_zero_points, n_clusters_k, order, cluster_assign_temp, codebook_temp, &actual_n_clusters_k)
		if actual_n_clusters_k == 0:
			mae_values.append(0.0)
			continue
		total_mae = 0.0
		for i in prange(num_non_zero_points, nogil=True):
			assigned_cluster_idx = cluster_assign_temp[i]
			centroid_x = codebook_temp[assigned_cluster_idx*2]
			centroid_y = codebook_temp[assigned_cluster_idx*2+1]
			point_mae = euclidean_distance_pts(&points[i*2], &codebook_temp[assigned_cluster_idx*2])
			total_mae += point_mae

		average_mae = total_mae / num_non_zero_points
		mae_values.append(average_mae)
		# print(f"Info: MAE for n_clusters = {n_clusters_k} is {average_mae}") # デバッグ出力
	cdef:
		int best_n_clusters = 1
		float max_distance_from_line = -1.0
		float x1, y1, x2, y2, m, c, current_n_float, current_mae, distance
		int n_clusters_eval
	if max_clusters_to_test > 1:
		x1 = 1.0
		y1 = mae_values[0]
		x2 = <float>max_clusters_to_test
		y2 = mae_values[max_clusters_to_test - 1]
		m = (y2 - y1) / (x2 - x1)
		c = y1 - m * x1
		for n_clusters_eval in range(2, max_clusters_to_test + 1):
			current_n_float = <float>n_clusters_eval
			current_mae = mae_values[n_clusters_eval - 1]
			distance = fabsf(m * current_n_float - current_mae + c) / sqrtf(m ** 2 + 1)
			if distance > max_distance_from_line:
				max_distance_from_line = distance
				best_n_clusters = n_clusters_eval
	# print(f"Info: Optimal number of clusters determined by Elbow method is {best_n_clusters}") # デバッグ出力
	for i in range(num_non_zero_points): order[i] = i
	cdef:
		float* final_codebook_ptr = <float*> malloc(best_n_clusters * 2 * sizeof(float))
		int[:, :, :] final_indices_view = np.zeros((H, W, 1), dtype=np.intc)
		int* final_cluster_assign = <int*> malloc(num_non_zero_points * sizeof(int))
	if not final_codebook_ptr or not final_cluster_assign:
		if points: free(points)
		if orig_i: free(orig_i)
		if orig_j: free(orig_j)
		if order: free(order)
		if cluster_assign_temp: free(cluster_assign_temp)
		if codebook_temp: free(codebook_temp)
		if final_codebook_ptr: free(final_codebook_ptr)
		if final_cluster_assign: free(final_cluster_assign)
		raise MemoryError("最終結果用配列のメモリ確保に失敗しました。")
	cdef int final_actual_n_clusters
	median_cut_clustering_internal(points, num_non_zero_points, best_n_clusters, order, final_cluster_assign, final_codebook_ptr, &final_actual_n_clusters)
	for i in prange(num_non_zero_points, nogil=True): final_indices_view[orig_i[i], orig_j[i], 0] = final_cluster_assign[i] + 1
	cdef:
		float[:] codebook_memoryview = <float[:final_actual_n_clusters * 2]>final_codebook_ptr
		np.ndarray final_codebook_np = np.asarray(codebook_memoryview).reshape((final_actual_n_clusters, 2))
	free(points)
	free(orig_i)
	free(orig_j)
	free(order)
	free(cluster_assign_temp)
	free(codebook_temp)
	free(final_cluster_assign)
	return final_codebook_np, final_indices_view
