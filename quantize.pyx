import numpy as np
cimport numpy as cnp
from cython.parallel import prange
def median_cut_rd(points: cnp.ndarray[cnp.float32_t], depth: int, lambda_rd: float, weights: cnp.ndarray = None):
	if weights is None: weights = np.array([1.0, 1.0])
	cdef float[2] weight = weights
	cdef int i, j
	cdef float[2] centroid, sqcent
	cdef int n = <int>points.shape[1]
	cdef float m = 1 / n
	cdef float[:, :] pt = points
	for i in prange(2):
		for j in prange(n):
			centroid[i] += pt[i, j]
			sqcent[i] += pt[i, j] ** 2
		centroid[i] *= m
		sqcent *= m

	cdef float distortion = (sqcent[0] - centroid[0] ** 2) * weight[0] + (sqcent[1] - centroid[1] ** 2) * weight[1]
	
	if depth == 0 or n <= 1: return [{'points': points, 'centroid': centroid, 'distortion': distortion}]

	p_min = np.min(points, axis=0)
	p_max = np.max(points, axis=0)
	weighted_range = weights * (p_max - p_min)
	# weighted_range[0] と weighted_range[1] を比較。等しければデフォルトで x 軸 (index 0) を選択
	axis = 0 if weighted_range[0] >= weighted_range[1] else 1
	
	# 選択された軸に沿って点をソート
	sorted_indices = np.argsort(points[:, axis])
	sorted_points = points[sorted_indices]
	
	# 中央付近で分割。ここでは単純に中央値で左右に分ける
	median_idx = sorted_points.shape[0] // 2
	left_cluster = sorted_points[:median_idx]
	right_cluster = sorted_points[median_idx:]
	
	# 再帰的に左右のクラスタについてメディアンカットを実行
	clusters_left = median_cut_rd(left_cluster, depth - 1, lambda_rd, weights)
	clusters_right = median_cut_rd(right_cluster, depth - 1, lambda_rd, weights)
	
	# 分割後の各クラスタから算出される総歪み（子クラスタの歪みの合計）
	new_clusters = clusters_left + clusters_right
	new_total_distortion = sum(cluster['distortion'] for cluster in new_clusters)
	
	# Rate–Distortion 最適化の判断:
	#   ・分割しない場合のコスト = 現在のクラスタの歪み
	#   ・分割する場合のコスト = (左右クラスタの歪みの合計) + lambda_rd（分割に伴う rate ペナルティ）
	cost_no_split = distortion
	cost_split = new_total_distortion + lambda_rd
	
	if cost_split < cost_no_split:
		# 分割したほうがトータルコストが低ければ分割結果を採用
		return new_clusters
	else:
		# 分割しないほうがよい場合、現在のクラスタをそのまま返す
		return [{'points': points, 'centroid': centroid, 'distortion': distortion}]
