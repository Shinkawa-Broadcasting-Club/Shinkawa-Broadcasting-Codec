# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3
# distutils: language=c

import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.stdlib cimport malloc, free

# 実用的な初期分割数
DEF INITIAL_BINS = 1024

# XORShift法の乱数生成のための関数
cdef inline int xorshift32(int state) nogil:
    cdef int result = state
    result ^= (result << 13)
    result ^= (result >> 17)
    result ^= (result << 5)
    return result

# 箱内の最小・最大を効率的に求める関数
cdef void find_min_max(int[:, :] data, int start, int end, int dim, int* min_val, int* max_val) nogil:
    cdef int i
    cdef int val1
    cdef int val2
    cdef int local_min = 2147483647  # INT_MAX
    cdef int local_max = -2147483648  # INT_MIN
    
    # ペアを作り、効率的に最小・最大を求める
    i = start
    while i + 1 < end:
        val1 = data[dim, i]
        val2 = data[dim, i+1]
        
        if val1 < val2:
            if val1 < local_min:
                local_min = val1
            if val2 > local_max:
                local_max = val2
        else:
            if val2 < local_min:
                local_min = val2
            if val1 > local_max:
                local_max = val1
        
        i += 2
    
    # 奇数個の場合、最後の要素を処理
    if i < end:
        val1 = data[dim, i]
        if val1 < local_min:
            local_min = val1
        if val1 > local_max:
            local_max = val1
    
    min_val[0] = local_min
    max_val[0] = local_max

# 箱内の分散を計算する関数（サンプリング使用）
cdef int calculate_variance(int[:, :] data, int start, int end, int dim, int* seed) nogil:
    cdef int i
    cdef int j
    cdef int val
    cdef int sample_count
    cdef int sum_val
    cdef int sum_squared
    cdef int variance = 0
    cdef int total_count = end - start
    cdef int sample_size = 256  # 最大サンプル数
    
    if total_count <= sample_size:
        # サンプル数が少ない場合は全てのデータを使用
        sum_val = 0
        sum_squared = 0
        
        for i in range(start, end):
            val = data[dim, i]
            sum_val += val
            sum_squared += val * val
        
        if total_count > 1:
            # 分散の計算 (整数値で返す)
            variance = (sum_squared - (sum_val * sum_val) / total_count) / (total_count - 1)
    else:
        # サンプリングを使用
        sum_val = 0
        sum_squared = 0
        sample_count = 0
        
        # XORShiftを使って無作為抽出
        for i in range(sample_size):
            seed[0] = xorshift32(seed[0])
            j = start + (seed[0] & 0x7FFFFFFF) % total_count  # 正の値を確保
            
            val = data[dim, j]
            sum_val += val
            sum_squared += val * val
            sample_count += 1
        
        if sample_count > 1:
            # 分散の計算 (整数値で返す)
            variance = (sum_squared - (sum_val * sum_val) / sample_count) / (sample_count - 1)
    
    return variance

# 箱の平均値を計算する関数
cdef int calculate_mean(int[:, :] data, int start, int end, int dim) nogil:
    cdef int i
    cdef int sum_val = 0
    cdef int count = end - start
    
    if count == 0:
        return 0
    
    for i in range(start, end):
        sum_val += data[dim, i]
    
    return sum_val // count

# 主要なメディアンカット関数の実装
cpdef tuple apply_median_cut(int[:, :] data, int variance_threshold):
    cdef int n_points = data.shape[1]  # データポイントの数
    cdef int n_dims = 6  # 6次元のデータ
    cdef int max_bins = 0x7FFFFFFF  # 最大分割数（理論上の最大値）
    cdef int practical_max_bins = min(n_points, 10000)  # 実用的な上限
    cdef cnp.ndarray[cnp.int32_t, ndim=2] labels
    cdef int[:, :] labels_view, codebook_view
    cdef cnp.ndarray[cnp.int32_t, ndim=2] codebook
    cdef int bin_count = 1  # 現在の箱の数
    cdef int current_allocated_bins = INITIAL_BINS  # 初期確保する箱の数
    cdef int** bin_start = NULL
    cdef int** bin_end = NULL
    cdef int* bin_status = NULL
    cdef int seed = 123456789
    cdef int i
    cdef int current_bin
    cdef int max_range_bin
    cdef int max_range_dim
    cdef int dim
    cdef int range_size
    cdef int max_range
    cdef int min_val
    cdef int max_val
    cdef int mean_val
    cdef int var_value
    cdef int left
    cdef int right
    cdef int temp_index
    cdef int j
    cdef int k
    cdef int need_more_memory
    cdef int** new_bin_start = NULL
    cdef int** new_bin_end = NULL
    cdef int* new_bin_status = NULL
    cdef int new_allocation_size
    
    # 出力配列の初期化
    labels = np.zeros((n_points,), dtype=np.int32).reshape(1, n_points)
    labels_view = labels
    
    # コードブックの初期化（各箱の代表値）- 初期化時は実用的な上限に制限
    codebook = np.zeros((practical_max_bins, n_dims), dtype=np.int32)
    codebook_view = codebook
    
    try:
        # 初期箱数分のメモリ確保
        bin_start = <int**>malloc(current_allocated_bins * sizeof(int*))
        bin_end = <int**>malloc(current_allocated_bins * sizeof(int*))
        bin_status = <int*>malloc(current_allocated_bins * sizeof(int))  # 0=未処理, 1=処理済み
        
        # メモリ確保の検証
        if bin_start == NULL or bin_end == NULL or bin_status == NULL:
            raise MemoryError("Failed to allocate initial memory for bin tracking")
        
        # 各箱の開始・終了インデックスの初期化
        for i in range(current_allocated_bins):
            bin_start[i] = <int*>malloc(sizeof(int))
            bin_end[i] = <int*>malloc(sizeof(int))
            if bin_start[i] == NULL or bin_end[i] == NULL:
                raise MemoryError("Failed to allocate memory for bin indices")
            bin_status[i] = 0
        
        # 最初の箱は全てのデータを含む
        bin_start[0][0] = 0
        bin_end[0][0] = n_points
        bin_status[0] = 0  # 未処理
        
        # メディアンカットの実行
        while bin_count < practical_max_bins:
            # もし現在の箱数が確保済みメモリ容量の80%を超えたら、メモリを拡張
            if bin_count > (current_allocated_bins * 4) // 5:
                need_more_memory = 1
                
                # 新しい確保サイズは現在の2倍（ただし最大値を超えないように）
                new_allocation_size = min(max_bins, current_allocated_bins * 2)
                
                if new_allocation_size <= current_allocated_bins:
                    # これ以上拡張できない
                    break
                
                # 新しいメモリを確保
                new_bin_start = <int**>malloc(new_allocation_size * sizeof(int*))
                new_bin_end = <int**>malloc(new_allocation_size * sizeof(int*))
                new_bin_status = <int*>malloc(new_allocation_size * sizeof(int))
                
                if new_bin_start == NULL or new_bin_end == NULL or new_bin_status == NULL:
                    # メモリ確保に失敗した場合は、現在のメモリで続行
                    need_more_memory = 0
                    if new_bin_start != NULL:
                        free(new_bin_start)
                        new_bin_start = NULL
                    if new_bin_end != NULL:
                        free(new_bin_end)
                        new_bin_end = NULL
                    if new_bin_status != NULL:
                        free(new_bin_status)
                        new_bin_status = NULL
                    
                    # 実用的な最大分割数を現在の箱数に制限
                    practical_max_bins = min(practical_max_bins, current_allocated_bins)
                else:
                    # 既存のデータを新しいメモリにコピー
                    for i in range(current_allocated_bins):
                        new_bin_start[i] = bin_start[i]
                        new_bin_end[i] = bin_end[i]
                        new_bin_status[i] = bin_status[i]
                    
                    # 新しいメモリ領域を初期化
                    for i in range(current_allocated_bins, new_allocation_size):
                        new_bin_start[i] = <int*>malloc(sizeof(int))
                        new_bin_end[i] = <int*>malloc(sizeof(int))
                        if new_bin_start[i] == NULL or new_bin_end[i] == NULL:
                            # 一部のメモリ確保に失敗した場合、確保できた分まで使う
                            new_allocation_size = i
                            break
                        new_bin_status[i] = 0
                    
                    # 古いメモリへのポインタを保存
                    free(bin_start)
                    free(bin_end)
                    free(bin_status)
                    
                    # 新しいメモリを使用
                    bin_start = new_bin_start
                    bin_end = new_bin_end
                    bin_status = new_bin_status
                    current_allocated_bins = new_allocation_size
            
            # 最大範囲を持つ箱を探す
            max_range = -1
            max_range_bin = -1
            max_range_dim = -1
            
            for current_bin in range(bin_count):
                if bin_status[current_bin] == 1:  # すでに処理済みの箱はスキップ
                    continue
                
                # この箱のサイズが小さすぎる場合は分割しない
                if bin_end[current_bin][0] - bin_start[current_bin][0] <= 1:
                    bin_status[current_bin] = 1  # 処理済みとしてマーク
                    continue
                
                # 各次元の範囲を計算し、最大のものを見つける
                for dim in range(n_dims):
                    find_min_max(data, bin_start[current_bin][0], bin_end[current_bin][0], dim, &min_val, &max_val)
                    range_size = max_val - min_val
                    
                    # 分散を計算
                    var_value = calculate_variance(data, bin_start[current_bin][0], bin_end[current_bin][0], dim, &seed)
                    
                    # 分散が閾値以下なら分割しない
                    if var_value < variance_threshold:
                        continue
                    
                    if range_size > max_range:
                        max_range = range_size
                        max_range_bin = current_bin
                        max_range_dim = dim
                
                # 全ての次元で分散が閾値以下なら、この箱は処理済みとしてマーク
                if max_range_bin == -1 or max_range_bin != current_bin:
                    bin_status[current_bin] = 1
            
            # 分割すべき箱がなければ終了
            if max_range_bin == -1:
                break
            
            # 次の分割で箱数が確保済みメモリを超える場合は終了
            if bin_count + 1 >= current_allocated_bins:
                break
            
            # 選ばれた箱を分割するための平均値を計算
            mean_val = calculate_mean(data, bin_start[max_range_bin][0], bin_end[max_range_bin][0], max_range_dim)
            
            # データを平均値で分割する（箱内でのソートなしに分割）
            left = bin_start[max_range_bin][0]
            right = bin_end[max_range_bin][0] - 1
            
            # 新しい箱を追加
            bin_start[bin_count][0] = bin_start[max_range_bin][0]
            bin_end[bin_count][0] = bin_end[max_range_bin][0]
            bin_status[bin_count] = 0
            
            # データポイントをin-placeで再編成し、平均値よりも小さいものを左側に、大きいものを右側に移動
            while left <= right:
                # 左側から平均値以上の要素を探す
                while left <= right and data[max_range_dim, left] < mean_val:
                    # この要素は左の箱に属する
                    labels_view[0, left] = max_range_bin
                    left += 1
                
                # 右側から平均値未満の要素を探す
                while left <= right and data[max_range_dim, right] >= mean_val:
                    # この要素は右の箱（新しい箱）に属する
                    labels_view[0, right] = bin_count
                    right -= 1
                
                # 左側と右側の要素を交換
                if left < right:
                    # 全ての次元でデータを交換
                    for dim in range(n_dims):
                        temp_index = data[dim, left]
                        data[dim, left] = data[dim, right]
                        data[dim, right] = temp_index
                    
                    # ラベルも交換
                    labels_view[0, left] = max_range_bin
                    labels_view[0, right] = bin_count
                    
                    left += 1
                    right -= 1
            
            # 箱の境界を更新
            bin_end[max_range_bin][0] = left
            bin_start[bin_count][0] = left
            
            # 両方の箱を未処理としてマーク
            bin_status[max_range_bin] = 0
            bin_status[bin_count] = 0
            
            # 箱の数を増やす
            bin_count += 1
        
        # 各箱の代表値（平均）を計算
        # もしコードブックの配列サイズが現在の箱数より小さい場合は再確保
        if bin_count > practical_max_bins:
            # 新しいコードブックを作成
            codebook = np.zeros((bin_count, n_dims), dtype=np.int32)
            codebook_view = codebook
        
        for i in range(bin_count):
            for dim in range(n_dims):
                codebook_view[i, dim] = calculate_mean(data, bin_start[i][0], bin_end[i][0], dim)
        
        # 残りのデータポイントにラベルを割り当て（まだ処理されていない場合）
        for i in range(n_points):
            if labels_view[0, i] >= bin_count:
                labels_view[0, i] = bin_count - 1
    
    finally:
        # メモリの解放
        if bin_start != NULL:
            for i in range(current_allocated_bins):
                if bin_start[i] != NULL:
                    free(bin_start[i])
            free(bin_start)
        
        if bin_end != NULL:
            for i in range(current_allocated_bins):
                if bin_end[i] != NULL:
                    free(bin_end[i])
            free(bin_end)
        
        if bin_status != NULL:
            free(bin_status)
    
    return labels, codebook[:bin_count]

cpdef inline median_cut(cnp.ndarray[cnp.int32_t, ndim=3] arr, int var):
    cdef int x = <int> arr.shape[0]
    cdef int y = <int> arr.shape[1]
    cdef int z = <int> arr.shape[2]
    cdef cnp.ndarray[cnp.int32_t, ndim=3] out = np.empty((x, y, z), np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] t0 = np.empty((6, (x >> 3) * (y >> 3) * (z >> 3)), np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] t1 = np.empty((120, (x >> 3) * (y >> 3) * (z >> 3)), np.int32), codebook
    cdef list cd_list = []
    cdef int[:, :, :] tmp = arr
    cdef int[:, :] tmp_in = t0
    cdef int[:, :] tmp_out = t1
    cdef int[120][3] combi = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 0, 6], [0, 0, 7], [0, 1, 1], [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 1, 6], [0, 1, 7], [0, 2, 2], [0, 2, 3], [0, 2, 4], [0, 2, 5], [0, 2, 6], [0, 2, 7], [0, 3, 3], [0, 3, 4], [0, 3, 5], [0, 3, 6], [0, 3, 7], [0, 4, 4], [0, 4, 5], [0, 4, 6], [0, 4, 7], [0, 5, 5], [0, 5, 6], [0, 5, 7], [0, 6, 6], [0, 6, 7], [0, 7, 7], [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4], [1, 1, 5], [1, 1, 6], [1, 1, 7], [1, 2, 2], [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6], [1, 2, 7], [1, 3, 3], [1, 3, 4], [1, 3, 5], [1, 3, 6], [1, 3, 7], [1, 4, 4], [1, 4, 5], [1, 4, 6], [1, 4, 7], [1, 5, 5], [1, 5, 6], [1, 5, 7], [1, 6, 6], [1, 6, 7], [1, 7, 7], [2, 2, 2], [2, 2, 3], [2, 2, 4], [2, 2, 5], [2, 2, 6], [2, 2, 7], [2, 3, 3], [2, 3, 4], [2, 3, 5], [2, 3, 6], [2, 3, 7], [2, 4, 4], [2, 4, 5], [2, 4, 6], [2, 4, 7], [2, 5, 5], [2, 5, 6], [2, 5, 7], [2, 6, 6], [2, 6, 7], [2, 7, 7], [3, 3, 3], [3, 3, 4], [3, 3, 5], [3, 3, 6], [3, 3, 7], [3, 4, 4], [3, 4, 5], [3, 4, 6], [3, 4, 7], [3, 5, 5], [3, 5, 6], [3, 5, 7], [3, 6, 6], [3, 6, 7], [3, 7, 7], [4, 4, 4], [4, 4, 5], [4, 4, 6], [4, 4, 7], [4, 5, 5], [4, 5, 6], [4, 5, 7], [4, 6, 6], [4, 6, 7], [4, 7, 7], [5, 5, 5], [5, 5, 6], [5, 5, 7], [5, 6, 6], [5, 6, 7], [5, 7, 7], [6, 6, 6], [6, 6, 7], [6, 7, 7], [7, 7, 7]]
    cdef int a, b, c, d, e
    for a in range(120):
        for b in prange(x >> 3, nogil=True, schedule='static'):
            for c in prange(y >> 3):
                for d in prange(z >> 3):
                    e = b * (y >> 3) * (z >> 3) + c * (z >> 3) + d
                    tmp_in[0, e] = tmp[b + combi[a][0], c + combi[a][1], d + combi[a][2]]
                    tmp_in[1, e] = tmp[b + combi[a][0], c + combi[a][2], d + combi[a][1]]
                    tmp_in[2, e] = tmp[b + combi[a][1], c + combi[a][0], d + combi[a][2]]
                    tmp_in[3, e] = tmp[b + combi[a][1], c + combi[a][2], d + combi[a][0]]
                    tmp_in[4, e] = tmp[b + combi[a][2], c + combi[a][0], d + combi[a][1]]
                    tmp_in[5, e] = tmp[b + combi[a][2], c + combi[a][1], d + combi[a][0]]
        t1[a], codebook = apply_median_cut(tmp_in, var)
        cd_list.append(codebook)
    return t1, cd_list
