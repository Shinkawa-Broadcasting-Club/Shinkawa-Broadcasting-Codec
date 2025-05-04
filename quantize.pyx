# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3

# Python側のインポートと Cython 用の型宣言
import numpy as np
cimport numpy as cnp

from cython.parallel import prange   # ※ reduction は使わずシンプルに for ループへ置換
cimport cython
from libc.stdlib cimport malloc, free

# モジュール直下に、float と unsigned int の再解釈用 union を定義
cdef union Converter:
    float f
    unsigned int i

###########################################
# LSD基数ソート (32bit float 用; 非負値のみ対応)
###########################################
cdef inline void lsd_radix_sort_slice(float[:, :] data, int* indices, int start, int end, int dim) nogil:
    cdef int n = end - start
    cdef int i, pass_, d
    cdef int bucket[256]
    cdef int bucketStart[256]
    cdef int digit
    cdef unsigned int ukey
    cdef Converter conv  # union 変数
    cdef int* temp = <int*> malloc(n * sizeof(int))
    if temp == NULL:
        with gil:
            raise MemoryError("Unable to allocate temporary memory in LSD sort")
        return
    for pass_ in range(4):  # 4 パスで 8bit ずつ処理
        for d in range(256):
            bucket[d] = 0
        for i in range(n):
            conv.f = data[dim, indices[start + i]]
            ukey = conv.i
            digit = (ukey >> (pass_ * 8)) & 0xFF
            bucket[digit] += 1
        bucketStart[0] = 0
        for d in range(1, 256):
            bucketStart[d] = bucketStart[d - 1] + bucket[d - 1]
        for i in range(n):
            conv.f = data[dim, indices[start + i]]
            ukey = conv.i
            digit = (ukey >> (pass_ * 8)) & 0xFF
            temp[bucketStart[digit]] = indices[start + i]
            bucketStart[digit] += 1
        for i in range(n):
            indices[start + i] = temp[i]
    free(temp)

###########################################
# 指定区間 [start, end) 内の各次元の最小値・最大値を計算
###########################################
cdef void compute_bounds(float[:, :] data, int* indices, int start, int end,
                         float* minvals, float* maxvals) nogil:
    cdef int d, i, idx
    cdef float val
    if start >= end:
        return
    idx = indices[start]
    for d in range(6):
        minvals[d] = data[d, idx]
        maxvals[d] = data[d, idx]
    for i in range(start + 1, end):
        idx = indices[i]
        for d in range(6):
            val = data[d, idx]
            if val < minvals[d]:
                minvals[d] = val
            if val > maxvals[d]:
                maxvals[d] = val

###########################################
# 箱内で最も広がりのある次元を選び、その次元の平均値を pivot として計算
###########################################
cdef int choose_split_dimension_and_pivot(float[:, :] data, int* indices, int start, int end, float* pivot) nogil:
    cdef float minvals[6], maxvals[6]
    cdef int d, best_dim = 0
    cdef float r, best_range = 0.0, sum_val = 0.0
    cdef int count = end - start, i
    compute_bounds(data, indices, start, end, minvals, maxvals)
    for d in range(6):
        r = maxvals[d] - minvals[d]
        if r > best_range:
            best_range = r
            best_dim = d
    for i in range(start, end):
        sum_val += data[best_dim, indices[i]]
    if count:
        pivot[0] = sum_val / count
    else:
        pivot[0] = 0.0
    return best_dim

###########################################
# 指定区間 [start, end) の箱を pivot に基づいて分割
###########################################
cdef void split_box(float[:, :] data, int* indices, int start, int end,
                    int* split_index, int* split_dim, float* pivot) nogil:
    cdef int d = choose_split_dimension_and_pivot(data, indices, start, end, pivot)
    split_dim[0] = d
    lsd_radix_sort_slice(data, indices, start, end, d)
    cdef int i, j = start
    cdef float val
    for i in range(start, end):
        val = data[d, indices[i]]
        if val > pivot[0]:
            break
        j = i + 1
    if j == start or j == end:
        j = start + (end - start) // 2
    split_index[0] = j

###########################################
# 箱内の各点の和を計算し、所属ラベルを設定する
# （シンプルな for ループにより和を計算）
###########################################
cdef void compute_box_stats(float[:, :] data, int* indices, int s, int e,
                              int box_label, int* labels, float* out_means) nogil:
    cdef int j, count = e - s, idx
    cdef float local_sum0 = 0.0, local_sum1 = 0.0, local_sum2 = 0.0
    cdef float local_sum3 = 0.0, local_sum4 = 0.0, local_sum5 = 0.0
    for j in range(s, e):
        idx = indices[j]
        local_sum0 += data[0, idx]
        local_sum1 += data[1, idx]
        local_sum2 += data[2, idx]
        local_sum3 += data[3, idx]
        local_sum4 += data[4, idx]
        local_sum5 += data[5, idx]
    for j in range(s, e):
        labels[indices[j]] = box_label
    if count:
        out_means[0] = local_sum0 / count
        out_means[1] = local_sum1 / count
        out_means[2] = local_sum2 / count
        out_means[3] = local_sum3 / count
        out_means[4] = local_sum4 / count
        out_means[5] = local_sum5 / count
    else:
        out_means[0] = out_means[1] = out_means[2] = out_means[3] = out_means[4] = out_means[5] = 0.0

###########################################
# エクスポート対象関数 (cdef public)
#
# 入力:
#   data: 6行×N列 の float 型メモリビュー
#   n_boxes: 生成したい箱（クラスタ）の個数
#
# 出力:
#   (labels, codebook)
#    labels:
#      ・分割数が256個以下 → unsigned char[:] (np.uint8)
#      ・257～65536個の場合 → unsigned short[:] (np.uint16)
#      ・それ以上の場合 → unsigned int[:] (np.uint32)
#    codebook: 各箱内の代表値 (各箱 6 要素のリストのリスト)
###########################################
cdef public object median_cut(float[:, :] data, int n_boxes):
    cdef int N = data.shape[1]
    cdef int i
    if N <= 0:
        raise ValueError("データには1点以上必要です。")
    cdef int* indices = <int*> malloc(N * sizeof(int))
    if indices == NULL:
        raise MemoryError("インデックス用メモリ割当に失敗")
    for i in range(N):
        indices[i] = i

    cdef list boxes = [(0, N)]
    cdef int chosen_box, s, e, new_split_index, split_d
    cdef float pivot_val[1]
    cdef float minvals[6], maxvals[6], local_range, range_diff, best_range
    cdef int box_index, d
    while len(boxes) < n_boxes:
        chosen_box = -1
        best_range = -1.0
        for box_index, (s, e) in enumerate(boxes):
            if e - s < 2:
                continue
            with nogil:
                compute_bounds(data, indices, s, e, minvals, maxvals)
            local_range = 0.0
            for d in range(6):
                range_diff = maxvals[d] - minvals[d]
                if range_diff > local_range:
                    local_range = range_diff
            if local_range > best_range:
                best_range = local_range
                chosen_box = box_index
        if chosen_box == -1:
            break
        s, e = boxes[chosen_box]
        with nogil:
            split_box(data, indices, s, e, &new_split_index, &split_d, pivot_val)
        boxes.pop(chosen_box)
        boxes.append((s, new_split_index))
        boxes.append((new_split_index, e))
    cdef int num_boxes = len(boxes)
    cdef int* labels = <int*> malloc(N * sizeof(int))
    if labels == NULL:
        free(indices)
        raise MemoryError("ラベル用メモリ割当に失敗")
    for i in range(N):
        labels[i] = -1
    cdef list codebook = [None] * num_boxes
    cdef float out_means[6]
    cdef int box_label, count, j, idx
    for box_label, (s, e) in enumerate(boxes):
        count = e - s
        with nogil:
            compute_box_stats(data, indices, s, e, box_label, labels, out_means)
        codebook[box_label] = [out_means[0], out_means[1], out_means[2],
                               out_means[3], out_means[4], out_means[5]]
    
    # 返却用ラベル配列用変数を if 分岐前に宣言
    cdef object ret_labels
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] labels_mv8 = None
    cdef cnp.ndarray[cnp.uint16_t, ndim=1] labels_mv16 = None
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] labels_mv32 = None

    if num_boxes <= 256:
        ret_labels = np.empty(N, dtype=np.uint8)
        labels_mv8 = ret_labels
        for i in range(N):
            labels_mv8[i] = <cnp.uint8_t> labels[i]
    elif num_boxes <= 65536:
        ret_labels = np.empty(N, dtype=np.uint16)
        labels_mv16 = ret_labels
        for i in range(N):
            labels_mv16[i] = <cnp.uint16_t> labels[i]
    else:
        ret_labels = np.empty(N, dtype=np.uint32)
        labels_mv32 = ret_labels
        for i in range(N):
            labels_mv32[i] = <cnp.uint32_t> labels[i]

    free(indices)
    free(labels)
    return ret_labels, codebook
