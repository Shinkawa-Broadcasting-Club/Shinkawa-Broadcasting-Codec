# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, embedsignature=True

import numpy as np
cimport numpy as cnp
from cython.parallel import prange

cdef extern from "stdlib.h" nogil:
    void* malloc(size_t size)
    void free(void* ptr)
    void* calloc(size_t nmemb, size_t size)

cdef extern from "string.h" nogil:
    void memcpy(void* dest, void* src, size_t n)

cdef extern from "limits.h" nogil:
    int INT_MAX
    int INT_MIN
    long long LLONG_MAX
    long long LLONG_MIN

cdef extern from "math.h" nogil:
    long long llabs(long long x)

cdef struct Box:
    int min_val[6]
    int max_val[6]
    int point_indices_start
    int point_indices_end
    int box_idx
    int representative_val[6]

cdef inline int integer_average(long long total_sum, int count) nogil:
    if count == 0: return 0
    return <int>(total_sum // count)

cdef inline long long integer_variance_numerator(const int* data, int count, int stride) nogil:
    cdef long long sum_val = 0
    cdef long long sum_sq = 0
    cdef int i
    cdef long long val
    for i in range(count):
        val = data[i * stride]
        sum_val += val
        sum_sq += val * val
    return <long long>count * sum_sq - sum_val * sum_val

cdef unsigned int xorshift_state = 123456789

cdef inline unsigned int xorshift() nogil:
    global xorshift_state
    cdef unsigned int x = xorshift_state
    x ^= (x << 13)
    x ^= (x >> 17)
    x ^= (x << 5)
    xorshift_state = x
    return x

cdef inline void calculate_min_max_pairwise(const int* data, int count, int stride, int* min_val, int* max_val) nogil:
    if count == 0:
        min_val[0] = 0
        max_val[0] = 0
        return
    cdef int i, j
    cdef int val1, val2
    cdef int current_min = data[0]
    cdef int current_max = data[0]
    if count % 2 != 0: i = 1
    else:
        i = 0
        val1 = data[0]
        val2 = data[stride]
        if val1 < val2:
            current_min = val1
            current_max = val2
        else:
            current_min = val2
            current_max = val1
        i = 2
    while i < count:
        val1 = data[i * stride]
        val2 = data[(i + 1) * stride]
        if val1 < val2:
            if val1 < current_min: current_min = val1
            if val2 > current_max: current_max = val2
        else:
            if val2 < current_min: current_min = val2
            if val1 > current_max: current_max = val1
        i += 2
    min_val[0] = current_min
    max_val[0] = current_max

cdef inline void lsd_radix_sort_1d(int* indices, const int* data, int count, int dimension, int num_dimensions) nogil:
    if count <= 1: return
    cdef int bits = 8
    cdef int max_val = 0
    cdef int min_val = 0
    cdef int i, j, shift, byte_idx
    cdef int* temp_indices = <int*> malloc(count * sizeof(int))
    cdef int* counts = <int*> calloc(1 << bits, sizeof(int))
    cdef int* starts = <int*> calloc(1 << bits, sizeof(int))
    cdef int val
    cdef unsigned int bucket
    if temp_indices is NULL or counts is NULL or starts is NULL:
        if temp_indices is not NULL: free(temp_indices)
        if counts is not NULL: free(counts)
        if starts is not NULL: free(starts)
        return
    calculate_min_max_pairwise(<const int*> &data[dimension], count, num_dimensions, &min_val, &max_val)
    cdef long long offset = -min_val
    for byte_idx in range(4):
        shift = byte_idx * bits
        for i in range(1 << bits): counts[i] = 0
        for i in range(count):
            val = data[indices[i] * num_dimensions + dimension]
            bucket = <unsigned int>((val + offset) >> shift) & ((1 << bits) - 1)
            counts[bucket] += 1
        starts[0] = 0
        for i in range(1, 1 << bits): starts[i] = starts[i - 1] + counts[i - 1]
        for i in range(count):
            val = data[indices[i] * num_dimensions + dimension]
            bucket = <unsigned int>((val + offset) >> shift) & ((1 << bits) - 1)
            temp_indices[starts[bucket]] = indices[i]
            starts[bucket] += 1
        memcpy(indices, temp_indices, count * sizeof(int))
    free(temp_indices)
    free(counts)
    free(starts)

cdef inline void median_cut_recursive_v2(int* points_flat, int* point_indices, int point_indices_start, int point_indices_end, int num_dimensions, Box* boxes, int* num_boxes, int max_boxes, long long variance_threshold_sq_scaled) nogil:
    cdef int num_points_in_box = point_indices_end - point_indices_start
    cdef int longest_dim = 0
    cdef int max_range = -1
    cdef int current_range
    cdef int box_min[6]
    cdef int box_max[6]
    cdef long long current_variance_numerator
    cdef int sample_count
    cdef int* sampled_values = NULL
    cdef int sampled_idx_in_box
    cdef int* dim_values = NULL
    cdef long long total_sum_longest_dim = 0
    cdef int split_value
    cdef int left_ptr = point_indices_start
    cdef int right_ptr = point_indices_end - 1
    cdef int temp_idx
    cdef int left_original_idx
    cdef int right_original_idx
    cdef int partition_idx
    cdef int current_box_idx
    cdef int original_point_idx
    cdef long long dim_sum
    cdef int i, j, k
    if num_boxes[0] >= max_boxes: return
    if num_points_in_box <= 1:
        current_box_idx = num_boxes[0]
        if current_box_idx < max_boxes:
            boxes[current_box_idx].point_indices_start = point_indices_start
            boxes[current_box_idx].point_indices_end = point_indices_end
            boxes[current_box_idx].box_idx = current_box_idx
            original_point_idx = point_indices[point_indices_start]
            for i in range(num_dimensions):
                boxes[current_box_idx].representative_val[i] = points_flat[original_point_idx * num_dimensions + i]
                boxes[current_box_idx].min_val[i] = boxes[current_box_idx].representative_val[i]
                boxes[current_box_idx].max_val[i] = boxes[current_box_idx].representative_val[i]
            num_boxes[0] += 1
        return
    longest_dim = 0
    max_range = -1
    for i in range(num_dimensions):
        dim_values = <int*> malloc(num_points_in_box * sizeof(int))
        if dim_values is NULL: return
        for j in range(num_points_in_box): dim_values[j] = points_flat[point_indices[point_indices_start + j] * num_dimensions + i]
        calculate_min_max_pairwise(dim_values, num_points_in_box, 1, &box_min[i], &box_max[i])
        free(dim_values)
        dim_values = NULL
        current_range = box_max[i] - box_min[i]
        if current_range > max_range:
            max_range = current_range
            longest_dim = i
    if num_points_in_box > 257:
        sample_count = 256
        sampled_values = <int*> malloc(sample_count * sizeof(int))
        if sampled_values is NULL: return
        for i in range(sample_count):
            sampled_idx_in_box = <int>(xorshift() % num_points_in_box)
            original_point_idx = point_indices[point_indices_start + sampled_idx_in_box]
            sampled_values[i] = points_flat[original_point_idx * num_dimensions + longest_dim]
        current_variance_numerator = integer_variance_numerator(sampled_values, sample_count, 1)
        free(sampled_values)
        sampled_values = NULL
    else:
        dim_values = <int*> malloc(num_points_in_box * sizeof(int))
        if dim_values is NULL: return
        for j in range(num_points_in_box):
            original_point_idx = point_indices[point_indices_start + j]
            dim_values[j] = points_flat[original_point_idx * num_dimensions + longest_dim]
        current_variance_numerator = integer_variance_numerator(dim_values, num_points_in_box, 1)
        free(dim_values)
        dim_values = NULL
    if current_variance_numerator < variance_threshold_sq_scaled:
        current_box_idx = num_boxes[0]
        if current_box_idx < max_boxes:
            boxes[current_box_idx].point_indices_start = point_indices_start
            boxes[current_box_idx].point_indices_end = point_indices_end
            boxes[current_box_idx].box_idx = current_box_idx
            for i in range(num_dimensions):
                dim_sum = 0
                for j in range(num_points_in_box):
                    original_point_idx = point_indices[point_indices_start + j]
                    dim_sum += points_flat[original_point_idx * num_dimensions + i]
                boxes[current_box_idx].representative_val[i] = integer_average(dim_sum, num_points_in_box)
            for i in range(num_dimensions):
                dim_values = <int*> malloc(num_points_in_box * sizeof(int))
                if dim_values is NULL: return
                for j in range(num_points_in_box):
                    original_point_idx = point_indices[point_indices_start + j]
                    dim_values[j] = points_flat[original_point_idx * num_dimensions + i]
                calculate_min_max_pairwise(dim_values, num_points_in_box, 1, &boxes[current_box_idx].min_val[i], &boxes[current_box_idx].max_val[i])
                free(dim_values)
                dim_values = NULL
            num_boxes[0] += 1
        return
    lsd_radix_sort_1d(&point_indices[point_indices_start], points_flat, num_points_in_box, longest_dim, num_dimensions)
    total_sum_longest_dim = 0
    for i in range(num_points_in_box):
        original_point_idx = point_indices[point_indices_start + i]
        total_sum_longest_dim += points_flat[original_point_idx * num_dimensions + longest_dim]
    split_value = integer_average(total_sum_longest_dim, num_points_in_box)
    left_ptr = point_indices_start
    right_ptr = point_indices_end - 1
    while left_ptr <= right_ptr:
        left_original_idx = point_indices[left_ptr]
        while points_flat[left_original_idx * num_dimensions + longest_dim] < split_value:
            left_ptr += 1
            if left_ptr > right_ptr: break
            left_original_idx = point_indices[left_ptr]
        right_original_idx = point_indices[right_ptr]
        while points_flat[right_original_idx * num_dimensions + longest_dim] >= split_value:
            right_ptr -= 1
            if left_ptr > right_ptr: break
            right_original_idx = point_indices[right_ptr]
        if left_ptr < right_ptr:
            temp_idx = point_indices[left_ptr]
            point_indices[left_ptr] = point_indices[right_ptr]
            point_indices[right_ptr] = temp_idx
            left_ptr += 1
            right_ptr -= 1
    partition_idx = left_ptr
    if partition_idx == point_indices_start or partition_idx == point_indices_end: partition_idx = point_indices_start + num_points_in_box // 2
    median_cut_recursive_v2(points_flat, point_indices, point_indices_start, partition_idx, num_dimensions, boxes, num_boxes, max_boxes, variance_threshold_sq_scaled)
    median_cut_recursive_v2(points_flat, point_indices, partition_idx, point_indices_end, num_dimensions, boxes, num_boxes, max_boxes, variance_threshold_sq_scaled)

cdef inline apply_median_cut(int[:, :] points_memview, int num_splits, int variance_threshold):
    cdef int num_dimensions = points_memview.shape[0]
    cdef int num_points = points_memview.shape[1]
    cdef int max_boxes
    cdef Box* boxes = NULL
    cdef int num_boxes_actual = 0
    cdef int* num_boxes_ptr = &num_boxes_actual
    cdef int* points_flat = NULL
    cdef int* original_indices = NULL
    cdef long long variance_threshold_sq_scaled
    cdef int i, j, k
    cdef int box_idx
    cdef int point_indices_start
    cdef int point_indices_end
    cdef int original_point_idx
    if num_dimensions != 6: raise ValueError("入力配列は6次元である必要があります (shape[0] == 6)")
    max_boxes = 1 << num_splits
    boxes = <Box*> calloc(max_boxes, sizeof(Box))
    if boxes is NULL: raise MemoryError("ボックスのメモリ割り当てに失敗しました")
    points_flat = <int*> malloc(num_points * num_dimensions * sizeof(int))
    if points_flat is NULL:
        free(boxes)
        raise MemoryError("フラット化された点のメモリ割り当てに失敗しました")
    for i in range(num_points):
        for j in range(num_dimensions): points_flat[i * num_dimensions + j] = points_memview[j, i]
    original_indices = <int*> malloc(num_points * sizeof(int))
    if original_indices is NULL:
        free(points_flat)
        free(boxes)
        raise MemoryError("元のインデックスのメモリ割り当てに失敗しました")
    for i in prange(num_points, schedule='static', nogil=True): original_indices[i] = i
    variance_threshold_sq_scaled = <long long>variance_threshold * 256 * 256
    median_cut_recursive_v2(points_flat, original_indices, 0, num_points, num_dimensions, boxes, num_boxes_ptr, max_boxes, variance_threshold_sq_scaled)
    cdef cnp.ndarray[int, ndim=1] box_indices_np = np.full(num_points, -1, dtype=np.intc)
    cdef cnp.ndarray[int, ndim=2] codebook_np = np.empty((num_boxes_actual, num_dimensions), dtype=np.intc)
    for i in range(num_boxes_actual):
        box_idx = boxes[i].box_idx
        point_indices_start = boxes[i].point_indices_start
        point_indices_end = boxes[i].point_indices_end
        for j in range(point_indices_start, point_indices_end):
             original_point_idx = original_indices[j]
             box_indices_np[original_point_idx] = box_idx
        for k in range(num_dimensions): codebook_np[box_idx, k] = boxes[i].representative_val[k]
    free(points_flat)
    free(original_indices)
    free(boxes)
    return box_indices_np, codebook_np

cpdef inline median_cut(cnp.ndarray[cnp.int32_t, ndim=3] arr):
    cdef int x = <int> arr.shape[0]
    cdef int y = <int> arr.shape[1]
    cdef int z = <int> arr.shape[2]
    cdef cnp.ndarray[cnp.int32_t, ndim=3] out = np.empty((x, y, z), np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] tmp_in = np.empty((6, (x >> 3) * (y >> 3) * (z >> 3)), np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] tmp_out = np.empty(((x >> 3) * (y >> 3) * (z >> 3)), np.int32)
    cdef int[:, :] tin
    cdef int[120][3] combi = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 0, 6], [0, 0, 7], [0, 1, 1], [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 1, 6], [0, 1, 7], [0, 2, 2], [0, 2, 3], [0, 2, 4], [0, 2, 5], [0, 2, 6], [0, 2, 7], [0, 3, 3], [0, 3, 4], [0, 3, 5], [0, 3, 6], [0, 3, 7], [0, 4, 4], [0, 4, 5], [0, 4, 6], [0, 4, 7], [0, 5, 5], [0, 5, 6], [0, 5, 7], [0, 6, 6], [0, 6, 7], [0, 7, 7], [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4], [1, 1, 5], [1, 1, 6], [1, 1, 7], [1, 2, 2], [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6], [1, 2, 7], [1, 3, 3], [1, 3, 4], [1, 3, 5], [1, 3, 6], [1, 3, 7], [1, 4, 4], [1, 4, 5], [1, 4, 6], [1, 4, 7], [1, 5, 5], [1, 5, 6], [1, 5, 7], [1, 6, 6], [1, 6, 7], [1, 7, 7], [2, 2, 2], [2, 2, 3], [2, 2, 4], [2, 2, 5], [2, 2, 6], [2, 2, 7], [2, 3, 3], [2, 3, 4], [2, 3, 5], [2, 3, 6], [2, 3, 7], [2, 4, 4], [2, 4, 5], [2, 4, 6], [2, 4, 7], [2, 5, 5], [2, 5, 6], [2, 5, 7], [2, 6, 6], [2, 6, 7], [2, 7, 7], [3, 3, 3], [3, 3, 4], [3, 3, 5], [3, 3, 6], [3, 3, 7], [3, 4, 4], [3, 4, 5], [3, 4, 6], [3, 4, 7], [3, 5, 5], [3, 5, 6], [3, 5, 7], [3, 6, 6], [3, 6, 7], [3, 7, 7], [4, 4, 4], [4, 4, 5], [4, 4, 6], [4, 4, 7], [4, 5, 5], [4, 5, 6], [4, 5, 7], [4, 6, 6], [4, 6, 7], [4, 7, 7], [5, 5, 5], [5, 5, 6], [5, 5, 7], [5, 6, 6], [5, 6, 7], [5, 7, 7], [6, 6, 6], [6, 6, 7], [6, 7, 7], [7, 7, 7]]
    cdef int a, b, c, d
    for a in range(120):
        for b in prange(x >> 3):
            for c in prange(y >> 3):
                for d in prange(z >> 3):
