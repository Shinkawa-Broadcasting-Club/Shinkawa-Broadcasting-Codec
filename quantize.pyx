# quantize.pyx

from cython.parallel import prange
cimport cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport sqrtf, fabsf

# 入出力は np.float16 を用いますが、内部計算は float (32bit) を使用
ctypedef np.float16_t F16

# ─────────────────────────────────────────
# SIMD 命令 (nmmintrin.h)
# ─────────────────────────────────────────
cdef extern from "nmmintrin.h":
    cdef struct __m128:
        pass
    cdef struct __m128i:
        pass
    __m128 _mm_setzero_ps()
    __m128 _mm_loadu_ps(const float* mem_addr)
    void _mm_storeu_ps(float* mem_addr, __m128 a)
    __m128 _mm_set1_ps(float w)
    __m128 _mm_sub_ps(__m128 a, __m128 b)
    __m128 _mm_add_ps(__m128 a, __m128 b)
    __m128 _mm_hadd_ps(__m128 a, __m128 b)
    __m128 _mm_and_ps(__m128 a, __m128 b)
    __m128i _mm_set1_epi32(int a)
    __m128 _mm_castsi128_ps(__m128i a)

# ─────────────────────────────────────────
# SIMD を用いた絶対誤差の和計算
# ─────────────────────────────────────────
cdef float simd_sum_abs_diff(const float* arr, int* indices, int start, int end, float cent) nogil:
    """
    指定区間 [start, end] の各要素について、|arr[indices[i]] - cent| の和を SIMD で求める。
    4要素ずつ並列計算し、余りは逐次計算。
    """
    cdef int i, n, simd_end
    cdef float s = 0.0
    n = end - start + 1
    simd_end = start + (n // 4) * 4
    cdef __m128 vsum = _mm_setzero_ps()
    cdef __m128 vcent = _mm_set1_ps(cent)
    cdef float tmp[4]
    for i in range(start, simd_end, 4):
        # gather の代替として一時バッファに 4 要素を格納
        tmp[0] = arr[ indices[i] ]
        tmp[1] = arr[ indices[i+1] ]
        tmp[2] = arr[ indices[i+2] ]
        tmp[3] = arr[ indices[i+3] ]
        cdef __m128 v = _mm_loadu_ps(&tmp[0])
        v = _mm_sub_ps(v, vcent)
        # 絶対値: 符号ビットをマスク (0x7fffffff)
        cdef __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff))
        v = _mm_and_ps(v, mask)
        vsum = _mm_add_ps(vsum, v)
    cdef float res[4]
    _mm_storeu_ps(&res[0], vsum)
    s = res[0] + res[1] + res[2] + res[3]
    for i in range(simd_end, start+n):
        s += fabsf(arr[ indices[i] ] - cent)
    return s

# ─────────────────────────────────────────
# クラスタ用構造体と統計計算関数
# ─────────────────────────────────────────
cdef struct Cluster:
    int start      # indices 配列上での開始位置
    int end        # indices 配列上での終了位置
    float min_x
    float max_x
    float min_y
    float max_y
    float sum_x
    float sum_y
    int count      # 含まれるデータ数
    int id         # クラスタ番号 (後の量子化ラベル用)

cdef void compute_cluster_stats(float* arr_x, float* arr_y, int* indices, int start, int end, Cluster* cluster):
    cdef int i
    cdef float x, y
    cluster.count = end - start + 1
    if cluster.count <= 0:
        return
    x = arr_x[ indices[start] ]
    y = arr_y[ indices[start] ]
    cluster.min_x = x; cluster.max_x = x
    cluster.min_y = y; cluster.max_y = y
    cluster.sum_x = x; cluster.sum_y = y
    for i in range(start+1, end+1):
        x = arr_x[ indices[i] ]
        y = arr_y[ indices[i] ]
        if x < cluster.min_x:
            cluster.min_x = x
        elif x > cluster.max_x:
            cluster.max_x = x
        if y < cluster.min_y:
            cluster.min_y = y
        elif y > cluster.max_y:
            cluster.max_y = y
        cluster.sum_x += x; cluster.sum_y += y

# ─────────────────────────────────────────
# 並列マージソートによるソートアルゴリズム
#
# 大規模データ向けに、まず小さい領域は逐次クイックソートし、ブロック毎に並列処理の後、段階的にマージする
# ─────────────────────────────────────────

# --- 基本的な整数のスワップ (nogil)
cdef void swap_int(int* a, int* b) nogil:
    cdef int tmp = a[0]
    a[0] = b[0]
    b[0] = tmp

# --- 逐次クイックソート (小領域向け)
cdef void sequential_quicksort(float* arr, int* indices, int low, int high) nogil:
    if low < high:
        cdef int pivotIndex = high
        cdef int pivotValIndex = indices[pivotIndex]
        cdef float pivot = arr[pivotValIndex]
        cdef int i = low - 1
        cdef int j
        for j in range(low, high):
            if arr[ indices[j] ] <= pivot:
                i += 1
                swap_int(&indices[i], &indices[j])
        swap_int(&indices[i+1], &indices[high])
        cdef int p = i + 1
        sequential_quicksort(arr, indices, low, p - 1)
        sequential_quicksort(arr, indices, p + 1, high)

# --- 2つのソート済み部分列をマージする関数
cdef void merge_two_sorted(float* arr, int* indices, int left, int mid, int right, int* temp) nogil:
    cdef int i = left, j = mid, k = left
    while i < mid and j < right:
        if arr[ indices[i] ] <= arr[ indices[j] ]:
            temp[k] = indices[i]
            i += 1
        else:
            temp[k] = indices[j]
            j += 1
        k += 1
    while i < mid:
        temp[k] = indices[i]
        i += 1; k += 1
    while j < right:
        temp[k] = indices[j]
        j += 1; k += 1

# --- 並列マージソート本体
cdef void parallel_merge_sort_indices(float* arr, int* indices, int n) nogil:
    cdef int block_size = 1024
    if block_size > n:
        block_size = n
    cdef int num_blocks = (n + block_size - 1) // block_size
    cdef int b
    for b in prange(num_blocks, schedule="static", nogil=True):
        cdef int start = b * block_size
        cdef int end = start + block_size - 1
        if end >= n:
            end = n - 1
        sequential_quicksort(arr, indices, start, end)
    cdef int* temp = <int*> malloc(n * sizeof(int))
    if temp == NULL:
        return
    cdef int width = block_size
    cdef int i, left, mid, right
    while width < n:
        for i in prange(0, n, 2*width, schedule="static", nogil=True):
            left = i
            mid = i + width
            right = i + 2*width
            if mid > n:
                mid = n
            if right > n:
                right = n
            merge_two_sorted(arr, indices, left, mid, right, temp)
        for i in prange(0, n, schedule="static", nogil=True):
            indices[i] = temp[i]
        width *= 2
    free(temp)

# --- ソート領域のサイズに応じたラッパー
cdef void sort_indices_range(float* arr, int* indices, int len) nogil:
    if len < 1024:
        sequential_quicksort(arr, indices, 0, len-1)
    else:
        parallel_merge_sort_indices(arr, indices, len)

# --- indices 配列の subarray [low, high] をソートするラッパー
cdef void sort_indices_subarray(float* arr, int* indices, int low, int high) nogil:
    cdef int len = high - low + 1
    sort_indices_range(arr, indices + low, len)

# ─────────────────────────────────────────
# 各クラスタに対して量子化後のクラスタインデックスを割り当てる
# ※ クラスタラベルが 1 から始まるよう i+1 としている
# ─────────────────────────────────────────
cdef void assign_quantized_indices(int* indices, Cluster* clusters, int num_clusters, int total_points, int* quantized) nogil:
    cdef int i, j
    for i in range(num_clusters):
        for j in range(clusters[i].start, clusters[i].end+1):
            quantized[ indices[j] ] = i + 1

# ─────────────────────────────────────────
# エントリポイント: Python から呼び出せる quantize_points()
#
# 入力:
#   data: np.ndarray(dtype=np.float16) で (N,2) の形状
#   t: 閥値 (F16)  --- 各点の (x,y) で、絶対値が t 未満ならフィルターして 0 とする
#
# 処理概要:
#   (1) フィルター: 閥値未満は除外、アクティブな点のみ処理
#   (2) アクティブ点のみでメディアンカット＋エルボー法によるクラスタリング・量子化
#   (3) 元の配列順序に戻し、フィルター点は 0 を、その他はクラスタラベル (1～) を出力
# ─────────────────────────────────────────
cpdef quantize_points(np.ndarray data, F16 t):
    cdef int i, j, n
    cdef int N = data.shape[0]
    if data.shape[1] != 2:
        raise ValueError("入力は (N,2) の形状である必要があります。")
    
    ###########################################
    # (1) フィルター処理：閥値 t 未満の点は除外
    ###########################################
    cdef float* act_x = <float*> malloc(N * sizeof(float))
    cdef float* act_y = <float*> malloc(N * sizeof(float))
    cdef int* act_orig_idx = <int*> malloc(N * sizeof(int))
    if act_x == NULL or act_y == NULL or act_orig_idx == NULL:
        if act_x: free(act_x)
        if act_y: free(act_y)
        if act_orig_idx: free(act_orig_idx)
        raise MemoryError()
    
    cdef int active_count = 0
    cdef float vx, vy
    for i in range(N):
        vx = <float> data[i, 0]
        vy = <float> data[i, 1]
        if fabsf(vx) < t and fabsf(vy) < t:
            continue
        else:
            act_x[active_count] = vx
            act_y[active_count] = vy
            act_orig_idx[active_count] = i
            active_count += 1
    if active_count == 0:
        free(act_x); free(act_y); free(act_orig_idx)
        cdef np.ndarray quantized_indices = np.zeros((N,), dtype=np.float16)
        cdef np.ndarray codebook = np.empty((0, 2), dtype=np.float16)
        return 0, codebook, quantized_indices
    cdef int M = active_count
    
    ###########################################
    # (2) アクティブ点のみでのクラスタリング
    ###########################################
    cdef int* base_indices = <int*> malloc(M * sizeof(int))
    if base_indices == NULL:
        free(act_x); free(act_y); free(act_orig_idx)
        raise MemoryError()
    for i in range(M):
        base_indices[i] = i

    cdef int max_n = 16
    if max_n > M:
        max_n = M
    cdef float errors[16]
    for n in range(1, max_n+1):
        cdef int* indices = <int*> malloc(M * sizeof(int))
        if indices == NULL:
            free(act_x); free(act_y); free(act_orig_idx); free(base_indices)
            raise MemoryError()
        for i in range(M):
            indices[i] = base_indices[i]
        cdef Cluster* clusters = <Cluster*> malloc(n * sizeof(Cluster))
        if clusters == NULL:
            free(act_x); free(act_y); free(act_orig_idx); free(base_indices); free(indices)
            raise MemoryError()
        clusters[0].start = 0
        clusters[0].end = M - 1
        compute_cluster_stats(act_x, act_y, indices, 0, M - 1, &clusters[0])
        clusters[0].id = 0
        cdef int current_clusters = 1
        while current_clusters < n:
            cdef int split_idx = -1
            cdef float max_range = -1.0
            cdef float range_x, range_y, cur_range
            for j in range(current_clusters):
                range_x = clusters[j].max_x - clusters[j].min_x
                range_y = clusters[j].max_y - clusters[j].min_y
                cur_range = range_x if range_x >= range_y else range_y
                if cur_range > max_range and clusters[j].count > 1:
                    max_range = cur_range
                    split_idx = j
            if split_idx == -1:
                break
            cdef int low = clusters[split_idx].start
            cdef int high = clusters[split_idx].end
            cdef bint use_x = ((clusters[split_idx].max_x - clusters[split_idx].min_x) >= (clusters[split_idx].max_y - clusters[split_idx].min_y))
            if use_x:
                sort_indices_subarray(act_x, indices, low, high)
            else:
                sort_indices_subarray(act_y, indices, low, high)
            cdef int mid = (low + high) // 2
            clusters[split_idx].end = mid
            compute_cluster_stats(act_x, act_y, indices, clusters[split_idx].start, clusters[split_idx].end, &clusters[split_idx])
            clusters[current_clusters].start = mid + 1
            clusters[current_clusters].end = high
            compute_cluster_stats(act_x, act_y, indices, clusters[current_clusters].start, clusters[current_clusters].end, &clusters[current_clusters])
            clusters[current_clusters].id = current_clusters
            current_clusters += 1
        cdef float centroids_x[16]
        cdef float centroids_y[16]
        for i in range(current_clusters):
            if clusters[i].count > 0:
                centroids_x[i] = clusters[i].sum_x / clusters[i].count
                centroids_y[i] = clusters[i].sum_y / clusters[i].count
            else:
                centroids_x[i] = 0.0
                centroids_y[i] = 0.0
        cdef float total_error = 0.0
        for j in range(current_clusters):
            total_error += simd_sum_abs_diff(act_x, indices, clusters[j].start, clusters[j].end, centroids_x[j])
            total_error += simd_sum_abs_diff(act_y, indices, clusters[j].start, clusters[j].end, centroids_y[j])
        errors[n-1] = total_error / M
        free(clusters)
        free(indices)
    
    cdef int best_n = 1
    cdef float max_distance = 0.0
    cdef float x1 = 1.0, y1 = errors[0]
    cdef float x2 = max_n, y2 = errors[max_n-1]
    cdef float dx = x2 - x1, dy = y2 - y1
    cdef float norm = sqrtf(dx*dx + dy*dy)
    if norm == 0:
        norm = 1.0
    for n in range(1, max_n+1):
        cdef float x0 = n, y0 = errors[n-1]
        cdef float dist = fabsf(dy*x0 - dx*y0 + x2*y1 - y2*x1) / norm
        if dist > max_distance:
            max_distance = dist
            best_n = n

    ###########################################
    # (3) best_n に対する最終クラスタリングと量子化
    ###########################################
    cdef int* final_indices = <int*> malloc(M * sizeof(int))
    if final_indices == NULL:
        free(act_x); free(act_y); free(act_orig_idx); free(base_indices)
        raise MemoryError()
    for i in range(M):
        final_indices[i] = base_indices[i]
    cdef Cluster* final_clusters = <Cluster*> malloc(best_n * sizeof(Cluster))
    if final_clusters == NULL:
        free(act_x); free(act_y); free(act_orig_idx); free(base_indices); free(final_indices)
        raise MemoryError()
    final_clusters[0].start = 0
    final_clusters[0].end = M - 1
    compute_cluster_stats(act_x, act_y, final_indices, 0, M - 1, &final_clusters[0])
    final_clusters[0].id = 0
    cdef int final_cluster_count = 1
    while final_cluster_count < best_n:
        cdef int split_idx = -1
        cdef float max_range = -1.0
        cdef float range_x, range_y, cur_range
        for i in range(final_cluster_count):
            range_x = final_clusters[i].max_x - final_clusters[i].min_x
            range_y = final_clusters[i].max_y - final_clusters[i].min_y
            cur_range = range_x if range_x >= range_y else range_y
            if cur_range > max_range and final_clusters[i].count > 1:
                max_range = cur_range
                split_idx = i
        if split_idx == -1:
            break
        cdef int low = final_clusters[split_idx].start
        cdef int high = final_clusters[split_idx].end
        cdef bint use_x = ((final_clusters[split_idx].max_x - final_clusters[split_idx].min_x) >= (final_clusters[split_idx].max_y - final_clusters[split_idx].min_y))
        if use_x:
            sort_indices_subarray(act_x, final_indices, low, high)
        else:
            sort_indices_subarray(act_y, final_indices, low, high)
        cdef int mid = (low + high) // 2
        final_clusters[split_idx].end = mid
        compute_cluster_stats(act_x, act_y, final_indices, final_clusters[split_idx].start, final_clusters[split_idx].end, &final_clusters[split_idx])
        final_clusters[final_cluster_count].start = mid + 1
        final_clusters[final_cluster_count].end = high
        compute_cluster_stats(act_x, act_y, final_indices, final_clusters[final_cluster_count].start, final_clusters[final_cluster_count].end, &final_clusters[final_cluster_count])
        final_clusters[final_cluster_count].id = final_cluster_count
        final_cluster_count += 1
        
    cdef float final_centroids_x[16]
    cdef float final_centroids_y[16]
    for i in range(final_cluster_count):
        if final_clusters[i].count > 0:
            final_centroids_x[i] = final_clusters[i].sum_x / final_clusters[i].count
            final_centroids_y[i] = final_clusters[i].sum_y / final_clusters[i].count
        else:
            final_centroids_x[i] = 0.0
            final_centroids_y[i] = 0.0
            
    cdef int* quantized_int = <int*> malloc(M * sizeof(int))
    if quantized_int == NULL:
        free(act_x); free(act_y); free(act_orig_idx); free(base_indices); free(final_indices); free(final_clusters)
        raise MemoryError()
    # assign_quantized_indices 内で i+1 により 1 からのラベルが設定される
    assign_quantized_indices(final_indices, final_clusters, final_cluster_count, M, quantized_int)
    
    ###########################################
    # (4) 結果出力：元の順序に戻す
    ###########################################
    cdef np.ndarray codebook = np.empty((final_cluster_count, 2), dtype=np.float16)
    for i in range(final_cluster_count):
        codebook[i, 0] = <F16> final_centroids_x[i]
        codebook[i, 1] = <F16> final_centroids_y[i]
    cdef np.ndarray quantized_indices = np.zeros((N,), dtype=np.float16)
    for i in range(M):
        quantized_indices[ act_orig_idx[i] ] = <F16> quantized_int[i]
    
    ###########################################
    # (5) メモリ解放
    ###########################################
    free(act_x)
    free(act_y)
    free(act_orig_idx)
    free(base_indices)
    free(final_indices)
    free(final_clusters)
    free(quantized_int)
    
    return best_n, codebook, quantized_indices
