# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False, profile=False, linetrace=False, c_api_binop_methods=False, c_string_type=bytes, c_string_encoding=ascii, embedsignature=True, optimize.inline_threshold=1000

import numpy as np
cimport numpy as cnp
from cython.parallel import prange

# SSE4.2 整数命令のインクルード
# 注意: MSVCの場合、これらの組み込み関数は通常 <intrin.h> を介して利用可能です。
# しかし、Cythonの cdef extern from "<smmintrin.h>" が標準です。
# MSVCのセットアップがこれらのヘッダーを見つけられるか、同等のMSVC組み込み関数を使用できるか確認してください。
cdef extern from "<smmintrin.h>" nogil:
    ctypedef long long __m128i

    # ロード/ストア
    __m128i _mm_loadu_si128(const __m128i *__P)
    void _mm_storeu_si128(__m128i *__P, __m128i __A)

    # 算術演算
    __m128i _mm_add_epi32(__m128i __A, __m128i __B)
    __m128i _mm_sub_epi32(__m128i __A, __m128i __B)
    __m128i _mm_mullo_epi32(__m128i __A, __m128i __B) # 32x32 -> 下位32ビット

    # 最小/最大
    __m128i _mm_min_epi32(__m128i __A, __m128i __B)
    __m128i _mm_max_epi32(__m128i __A, __m128i __B)

    # セット/ゼロ
    __m128i _mm_set1_epi32(int __A)
    __m128i _mm_setzero_si128()

    # シャッフル (SSE2)
    __m128i _mm_shuffle_epi32(__m128i __A, int __imm8)

# SSE2 組み込み関数 (64ビット加算および32ビット乗算で64ビット結果を得るため) のインクルード
cdef extern from "<emmintrin.h>" nogil:
    __m128i _mm_add_epi64(__m128i __A, __m128i __B)
    __m128i _mm_mul_epi32(__m128i __A, __m128i __B) # SSE2: 32x32 -> 64ビット (2つのペアの下位半分と上位半分)
                                                   # 二乗の64ビット積を得るために使用

# 分散サンプリングのためのシンプルなXORShift乱数生成器
cdef unsigned int xorshift_state = 123456789

cdef inline unsigned int xorshift() nogil:
    # xorshift_state をグローバルとして宣言し、変更できるようにする
    global xorshift_state
    cdef unsigned int x = xorshift_state
    x ^= (x << 13)
    x ^= (x >> 17)
    x ^= (x << 5)
    xorshift_state = x
    return x

# 箱を表す構造体
cdef struct Box:
    int start_idx # 平坦化されたデータ配列内の開始インデックス
    int end_idx   # 平坦化されたデータ配列内の終了インデックス (排他的)
    int box_idx   # 最終コードブック内のこの箱のインデックス

# SIMDを使用して箱内の指定された次元の最小/最大を計算するヘルパー関数
# 入力データメモリビューがF-contiguousであり、SIMDロードによる効率的な列アクセスが可能であることを前提とします。
cdef inline void calculate_min_max_simd(int[:, :] data, int start_idx, int end_idx, int dim, int* min_val, int* max_val) nogil:
    cdef int n = end_idx - start_idx
    if n <= 0:
        min_val[0] = 0 # 空の箱のデフォルト値
        max_val[0] = 0 # 空の箱のデフォルト値
        return

    cdef int i
    cdef __m128i current_min_v, current_max_v, data_v

    # 指定された次元と範囲のデータの開始へのポインタを取得
    # これはメモリビューがF-contiguousの場合に有効です。
    cdef int* dim_data_ptr = &data[dim, start_idx]

    # 最初の4要素以下で最小/最大を初期化
    if n >= 4:
        current_min_v = _mm_loadu_si128(<__m128i*>(dim_data_ptr))
        current_max_v = _mm_loadu_si128(<__m128i*>(dim_data_ptr))
        i = start_idx + 4
    else:
        # 4要素未満の場合、最初の要素を手動で処理
        min_val[0] = data[dim, start_idx]
        max_val[0] = data[dim, start_idx]
        for i in range(start_idx + 1, end_idx):
            if data[dim, i] < min_val[0]:
                min_val[0] = data[dim, i]
            if data[dim, i] > max_val[0]:
                max_val[0] = data[dim, i]
        return # 全て手動で処理された場合は終了

    # SIMDを使用して4要素ずつ処理
    while i <= end_idx - 4:
        dim_data_ptr = &data[dim, i]
        data_v = _mm_loadu_si128(<__m128i*>(dim_data_ptr))
        current_min_v = _mm_min_epi32(current_min_v, data_v)
        current_max_v = _mm_max_epi32(current_max_v, data_v)
        i += 4

    # 残りの要素 (4未満) を手動で処理
    while i < end_idx:
         current_min_v = _mm_min_epi32(current_min_v, _mm_set1_epi32(data[dim, i]))
         current_max_v = _mm_max_epi32(current_max_v, _mm_set1_epi32(data[dim, i]))
         i += 1

    # 最終SIMDレジスタの水平最小/最大削減
    cdef int min_arr[4], max_arr[4]
    _mm_storeu_si128(<__m128i*>min_arr, current_min_v)
    _mm_storeu_si128(<__m128i*>max_arr, current_max_v)

    min_val[0] = min_arr[0]
    for i in range(1, 4):
        if min_arr[i] < min_val[0]:
            min_val[0] = min_arr[i]

    max_val[0] = max_arr[0]
    for i in range(1, 4):
        if max_arr[i] > max_val[0]:
            max_val[0] = max_arr[i]


# SIMDを使用して箱内の指定された次元の平均を計算するヘルパー関数
# 入力データメモリビューがF-contiguousであり、SIMDロードによる効率的な列アクセスが可能であることを前提とします。
cdef inline int calculate_average_simd(int[:, :] data, int start_idx, int end_idx, int dim) nogil:
    cdef int n = end_idx - start_idx
    if n <= 0:
        return 0

    cdef int i
    cdef __m128i sum_v = _mm_setzero_si128() # 4要素の合計
    cdef __m128i data_v

    # 指定された次元と範囲のデータの開始へのポインタを取得
    # これはメモリビューがF-contiguousの場合に有効です。
    cdef int* dim_data_ptr = &data[dim, start_idx]

    # SIMDを使用して4要素ずつ処理
    for i in range(start_idx, end_idx - 3, 4):
        dim_data_ptr = &data[dim, i]
        data_v = _mm_loadu_si128(<__m128i*>(dim_data_ptr))
        sum_v = _mm_add_epi32(sum_v, data_v)

    # 残りの要素 (4未満) を手動で処理
    cdef int remaining_sum = 0
    for i in range(start_idx + (end_idx - start_idx) // 4 * 4, end_idx):
        remaining_sum += data[dim, i]

    # 最終SIMDレジスタの水平合計削減
    cdef int sum_arr[4]
    _mm_storeu_si128(<__m128i*>sum_arr, sum_v)
    cdef long long total_sum = <long long>sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] + remaining_sum

    # 平均値の整数除算
    return <int>(total_sum // n)


# SIMDを使用して箱内の指定された次元の分散を計算するヘルパー関数 (サンプリングあり)
# 入力データメモリビューがF-contiguousであり、サンプリングされたデータの手動ロードによる効率的な列アクセスが可能であることを前提とします。
cdef inline int calculate_variance_simd(int[:, :] data, int start_idx, int end_idx, int dim, int average) nogil:
    cdef int n = end_idx - start_idx
    if n <= 1: # 0または1点の箱の分散は0
        return 0

    cdef int i
    # オーバーフローを防ぐために、差の二乗の合計に64ビット累積を使用
    cdef __m128i sum_sq_diff_v_low = _mm_setzero_si128() # ペア0, 2の (差*差)_64 を累積
    cdef __m128i sum_sq_diff_v_high = _mm_setzero_si128() # ペア1, 3の (差*差)_64 を累積

    cdef __m128i data_v, avg_v, diff_v, sq_diff_v_low64, sq_diff_v_high64
    cdef int count = 0
    cdef int limit = min(n, 256) # 要件に応じたサンプルサイズ

    avg_v = _mm_set1_epi32(average)

    # XORShiftを使用してインデックスをサンプリング
    cdef int sampled_indices[256] # 最大サンプルサイズ
    cdef int current_idx
    cdef int temp_data[4]
    cdef __m128i diff_v_shuffled

    if n <= 256:
        # サンプルサイズが箱のサイズ以上の場合、全ての点を使用
        for i in range(n):
            sampled_indices[i] = start_idx + i
        limit = n
    else:
        # XORShiftを使用して256点をサンプリング
        # xorshift() がGILを必要とする場合、このループは nogil ブロックの外にある必要がありますが、
        # xorshift() は nogil で宣言されているため、問題ありません。
        for i in range(256):
            sampled_indices[i] = start_idx + (xorshift() % n)
        limit = 256

    # 手動ロードとSIMDを使用して、サンプリングされたインデックスを4つずつバッチ処理
    for i in range(0, limit - 3, 4):
        # 4つのサンプリングされた値を手動で一時配列にロードし、その後SIMDレジスタにロード
        temp_data[0] = data[dim, sampled_indices[i]]
        temp_data[1] = data[dim, sampled_indices[i+1]]
        temp_data[2] = data[dim, sampled_indices[i+2]]
        temp_data[3] = data[dim, sampled_indices[i+3]]
        data_v = _mm_loadu_si128(<__m128i*>temp_data)

        # 平均との差を計算
        diff_v = _mm_sub_epi32(data_v, avg_v)

        # 32x32 -> 64ビット乗算 (SSE2 _mm_mul_epi32) を使用して二乗差を計算
        # _mm_mul_epi32(a, b) は (a[0]*b[0])_64 と (a[2]*b[2])_64 を計算します
        sq_diff_v_low64 = _mm_mul_epi32(diff_v, diff_v) # [(diff_v[0]*diff_v[0])_64, (diff_v[2]*diff_v[2])_64]

        # (diff_v[1]*diff_v[1])_64 と (diff_v[3]*diff_v[3])_64 が必要です
        # _mm_mul_epi32 のために要素1と3を整列させるために diff_v をシャッフル
        # _mm_shuffle_epi32(diff_v, _MM_SHUFFLE(w, z, y, x)) -> result[0]=p[x], result[1]=p[y], result[2]=p[z], result[3]=p[w]
        # [diff_v[1], diff_v[0], diff_v[3], diff_v[2]] を得るには _MM_SHUFFLE(2, 3, 0, 1) を使用
        diff_v_shuffled = _mm_shuffle_epi32(diff_v, 0b10110001) # [diff_v[1], diff_v[0], diff_v[3], diff_v[2]]
        sq_diff_v_high64 = _mm_mul_epi32(diff_v_shuffled, diff_v_shuffled) # [(diff_v[1]*diff_v[1])_64, (diff_v[3]*diff_v[3])_64]

        # SSE2 _mm_add_epi64 を使用して64ビット二乗差を合計
        sum_sq_diff_v_low = _mm_add_epi64(sum_sq_diff_v_low, sq_diff_v_low64)
        sum_sq_diff_v_high = _mm_add_epi64(sum_sq_diff_v_high, sq_diff_v_high64)

        count += 4

    # 残りのサンプリングされた要素 (4未満) を手動で処理
    cdef long long remaining_sq_diff_sum = 0
    cdef long long diff
    for i in range(count, limit):
        current_idx = sampled_indices[i]
        diff = <long long>data[dim, current_idx] - average
        remaining_sq_diff_sum += diff * diff

    # 64ビット合計の水平合計削減
    cdef long long sum_sq_diff_arr_low[2], sum_sq_diff_arr_high[2]
    _mm_storeu_si128(<__m128i*>sum_sq_diff_arr_low, sum_sq_diff_v_low)
    _mm_storeu_si128(<__m128i*>sum_sq_diff_arr_high, sum_sq_diff_v_high)

    cdef long long total_sq_diff_sum = sum_sq_diff_arr_low[0] + sum_sq_diff_arr_low[1] + \
                                       sum_sq_diff_arr_high[0] + sum_sq_diff_arr_high[1] + \
                                       remaining_sq_diff_sum

    # 分散の整数除算
    # 除算にはサンプルサイズ (limit) を使用
    if limit <= 0: return 0
    return <int>(total_sq_diff_sum // limit)


# 指定された次元に沿った分割値に基づいて箱内の点を分割するヘルパー関数
# 第2パーティションの最初の要素のインデックスを返します
# この関数は data 配列と box_indices 配列をインプレースで変更します。
cdef inline int partition_by_value(int[:, :] data, int[:] box_indices, int start_idx, int end_idx, int dim, int split_value) nogil:
    cdef int left = start_idx
    cdef int right = end_idx - 1
    cdef int i
    cdef int temp_val
    cdef int temp_idx

    # data[dim, i] < split_value の点が左側に、>= split_value の点が右側になるように点を分割
    while left <= right:
        # split_value 以上の左側の要素を見つける
        while left <= right and data[dim, left] < split_value:
            left += 1
        # split_value 未満の右側の要素を見つける
        while left <= right and data[dim, right] >= split_value:
            right -= 1

        # ポインタが交差していない場合、要素を交換 (box_indicesを含む完全な点)
        if left < right:
            # データ列を交換 (6つの特徴)
            for i in range(6):
                temp_val = data[i, left]
                data[i, left] = data[i, right]
                data[i, right] = temp_val

            # 箱インデックスを交換
            temp_idx = box_indices[left]
            box_indices[left] = box_indices[right]
            box_indices[right] = temp_idx

            left += 1
            right -= 1 # ここを right -= 1 に修正

    # 'left' は第2パーティション (>= split_value) の最初の要素のインデックスです
    return left

# コアとなるメディアンカット関数
cdef inline apply_median_cut(int[:, :] data, int num_target_boxes):
    """
    入力データにメディアンカットアルゴリズムを適用します。

    最大の範囲を持つ次元に沿った平均値に基づいて箱を分割します。
    計算には整数演算とSIMD (SSE4.2) を使用します。

    Args:
        data (int[:, :]): (6, N) の形状を持つ入力データメモリビュー。Nは点の数で、各点は6つの整数特徴を持ちます。
                          効率的なSIMDロードのためにF-contiguousであることを前提とします。
        num_target_boxes (int): 目標とする箱 (色) の数。

    Returns:
        tuple: 以下の要素を含むタプル:
            - box_indices (int[:]): (N,) の形状を持つ配列。各要素は対応する点が属する箱のインデックスです。
                                     インデックスは0から num_final_boxes - 1 までの連続した値です。
            - codebook (int[:, :]): (num_final_boxes, 6) の形状を持つ配列。
                                    各最終箱の代表的な平均値を含みます。
    """
    cdef int N = data.shape[1]
    cdef int num_dims = data.shape[0] # 6であるはず

    if num_dims != 6:
        # これは主に安全のためのチェックであり、ラッパーは (6, N) を提供するはずです
        raise ValueError("入力データは点ごとに6つの次元を持つ必要があります。")

    # 各点の箱インデックスを格納する配列
    # この配列は分割中に並べ替えられます
    cdef cnp.ndarray[cnp.int32_t, ndim=1] box_indices_arr = np.empty(N, dtype=np.int32)
    cdef int[:] box_indices = box_indices_arr

    # 最初は全ての点が箱0に属する
    with nogil:
        for i in range(N):
            box_indices[i] = 0

    # 処理対象の箱を管理するためのリスト (Box構造体として)
    cdef list box_queue = []
    box_queue.append(Box(0, N, 0)) # 全ての点を含む初期の箱を追加

    cdef list final_boxes = [] # これ以上分割されない最終的な箱を格納

    cdef int current_box_idx_counter = 0 # 一意な箱インデックスを割り当てるためのカウンター
    cdef Box current_box
    cdef int start_idx, end_idx, n
    cdef int min_vals[6], max_vals[6], ranges[6]
    cdef int max_range = -1
    cdef int split_dim = -1
    cdef int average_val
    cdef int variance_val
    cdef int split_point_idx
    cdef int box1_idx
    cdef int box2_idx

    # 分割のための分散閾値 (例の値、データに合わせて調整が必要)
    # 閾値が高いほど、特定の分散に対して分割が少なくなります。
    cdef int VARIANCE_THRESHOLD = 1000 # 例の閾値

    # 目標の箱数に達するか、それ以上分割できない箱がなくなるまで箱を処理
    # この条件は、キュー内の現在の箱を最終箱に追加した場合に目標を満たすか超えるかを確認します。
    # その場合、分割を停止します。
    while len(box_queue) > 0 and (len(final_boxes) + len(box_queue)) < num_target_boxes:
        # キューから次の箱を取得
        current_box = box_queue.pop(0)
        start_idx = current_box.start_idx
        end_idx = current_box.end_idx
        n = end_idx - start_idx

        if n <= 1: # 0または1点の箱は分割できない
            final_boxes.append(current_box)
            continue

        # 現在の箱内で最も範囲の大きい次元を見つける
        max_range = -1
        split_dim = -1
        # このループは6つの次元に対して順次実行されます
        for dim in range(num_dims):
            # SIMDを使用して現在の箱内のこの次元の最小/最大を計算
            calculate_min_max_simd(data, start_idx, end_idx, dim, &min_vals[dim], &max_vals[dim])
            ranges[dim] = max_vals[dim] - min_vals[dim]
            if ranges[dim] > max_range:
                max_range = ranges[dim]
                split_dim = dim

        # max_range が0の場合、この箱内の全ての点は全ての次元で同一であり、分割できない
        if max_range == 0:
            final_boxes.append(current_box)
            continue

        # SIMDを使用して分割次元に沿った平均値を計算
        average_val = calculate_average_simd(data, start_idx, end_idx, split_dim)

        # SIMDを使用して分割次元に沿った分散を計算 (必要に応じてサンプリングあり)
        variance_val = calculate_variance_simd(data, start_idx, end_idx, split_dim, average_val)

        # 分散と目標の箱数に基づいて分割するかどうかを決定
        # 分散が十分に高く、かつまだ箱が必要な場合に分割
        # 分割を検討するには、少なくとも2つのスロット (現在の箱が2つになる) が必要です。
        if variance_val > VARIANCE_THRESHOLD and (len(final_boxes) + len(box_queue) + 1) < num_target_boxes:
             # 分割次元に沿った平均値に基づいて点を分割
             # 分割関数は data 配列と box_indices 配列をインプレースで変更します
             # 第2パーティションの最初の要素のインデックスを返します。
             split_point_idx = partition_by_value(data, box_indices, start_idx, end_idx, split_dim, average_val)

             # 両方のパーティションに少なくとも1つの点があることを確認
             # If split_point_idx is the start or end index, partitioning didn't effectively split.
             # 分割が効果的に行われなかった場合 (例: 全ての値が平均以下、または全て平均以上)、
             # このイテレーションではこの箱を分割しない。
             if split_point_idx == start_idx or split_point_idx == end_idx:
                 final_boxes.append(current_box)
                 continue

             # Create two new boxes from the partitioned range
             # Assign new unique box indices
             current_box_idx_counter += 1
             box1_idx = current_box_idx_counter
             box_queue.append(Box(start_idx, split_point_idx, box1_idx))

             current_box_idx_counter += 1
             box2_idx = current_box_idx_counter
             box_queue.append(Box(split_point_idx, end_idx, box2_idx))

             # 2つの新しい箱の点の箱インデックスを更新
             # [start_idx, split_point_idx) の点は box1_idx を取得
             # [split_point_idx, end_idx) の点は box2_idx を取得
             # この更新は分割によって点が並べ替えられた後に行われます。
             with nogil:
                 for i in range(start_idx, split_point_idx):
                     box_indices[i] = box1_idx
                 for i in range(split_point_idx, end_idx):
                     box_indices[i] = box2_idx

        else:
            # 分割しない場合、最終箱に追加
            final_boxes.append(current_box)

    # キューに残っている箱を全て最終箱に追加 (目標に達したか、それ以上分割できない場合)
    final_boxes.extend(box_queue)

    # 最終的な箱インデックスが0から num_final_boxes - 1 までの連続した値であることを保証
    # 連続していない可能性のある box_idx の値から連続したインデックスへのマッピングを作成。
    # この部分はPython辞書を使用するため、GILが必要です。
    cdef dict old_to_new_box_idx_map = {}
    cdef int final_box_count = 0
    cdef Box final_box
    for final_box in final_boxes:
        old_to_new_box_idx_map[final_box.box_idx] = final_box_count
        final_box.box_idx = final_box_count # コードブック生成のために構造体内の box_idx を更新
        final_box_count += 1

    # 新しい連続したインデックスでメインの box_indices 配列を更新
    # このループはPython辞書へのアクセスを必要とするため、nogil ブロックの外にある必要があります。
    for i in range(N):
        box_indices[i] = old_to_new_box_idx_map[box_indices[i]]


    # コードブックを計算 (各最終箱の代表値)
    # 代表値は箱内の点の平均値です。
    cdef cnp.ndarray[cnp.int32_t, ndim=2] codebook_arr = np.empty((final_box_count, num_dims), dtype=np.int32)
    cdef int[:, :] codebook = codebook_arr

    cdef int box_point_count
    cdef long long dim_sum
    cdef int point_idx

    # 最終箱を繰り返し処理し、各次元の平均を計算
    for final_box in final_boxes:
        start_idx = final_box.start_idx
        end_idx = final_box.end_idx
        box_idx = final_box.box_idx
        box_point_count = end_idx - start_idx

        if box_point_count > 0:
            # この箱内の各次元の平均を計算
            # このループはデータアクセスが安全であれば nogil で実行可能。
            with nogil:
                for dim in range(num_dims):
                    dim_sum = 0
                    # この次元の合計を手動で計算 (C-contiguousとF-contiguousの両方で機能)
                    # データがF-contiguousで box_point_count が大きい場合、ここでSIMDを使用することも可能。
                    # 手動合計の方がシンプルで一般的に安全。
                    for point_idx in range(start_idx, end_idx):
                        dim_sum += data[dim, point_idx]

                    # 平均を計算 (整数除算)
                    codebook[box_idx, dim] = <int>(dim_sum // box_point_count)
        else:
            # 空の箱を処理 (N > 0 の場合、現在のロジックでは発生しないはず)
            with nogil:
                for dim in range(num_dims):
                    codebook[box_idx, dim] = 0 # 空の箱のデフォルト値

    # 箱インデックスとコードブックを返す
    return box_indices_arr, codebook_arr


# 提供されたラッパー関数を適合
# このラッパーは、3D配列からのデータをコアアルゴリズム用の (6, N) 形式に準備します。
# データ準備ロジックは独特で、'combi' 配列によって定義されたブロックインデックスの順列に基づいて
# 8x8x8ブロックごとに6つの特徴を抽出します。
cpdef inline median_cut(cnp.ndarray[cnp.int32_t, ndim=3] arr):
    cdef int x = <int> arr.shape[0]
    cdef int y = <int> arr.shape[1]
    cdef int z = <int> arr.shape[2]

    # 各次元の8x8x8ブロック数を計算
    cdef int num_blocks_x = x >> 3
    cdef int num_blocks_y = y >> 3
    cdef int num_blocks_z = z >> 3

    # 処理される点の総数、すなわちブロックの総数を計算
    cdef int total_points = num_blocks_x * num_blocks_y * num_blocks_z

    # apply_median_cut のための入力データ配列を準備。
    # 形状は (6, total_points) で、各列は6つの特徴を持つ点です。
    # この配列は順列ごとに異なる方法で埋められます。
    # F-contiguous (Fortran-contiguous) な配列として明示的に作成
    cdef cnp.ndarray[cnp.int32_t, ndim=2] tmp_in_arr = np.empty((6, total_points), dtype=np.int32, order='F')
    # メモリビューも F-contiguous モードで宣言
    cdef int[:, :], mode='fortran' tmp_in
    tmp_in = tmp_in_arr

    # 元のラッパーの出力 `t1` は (120, total_points) の形状を持ちます。
    # これは、120個の順列それぞれに対してメディアンカットを適用した結果の箱インデックスを格納していることを示唆しています。
    cdef cnp.ndarray[cnp.int32_t, ndim=2] t1_arr = np.empty((120, total_points), dtype=np.int32)
    cdef int[:, :] t1 = t1_arr

    # 順列ごとに生成されたコードブックを格納するリスト
    cdef list cd_list = []

    # 提供された combi 配列。これは各点 (ブロック) に対して特徴がどのように抽出されるかを定義します。
    # インデックス [0-7] はブロックインデックス [b, c, d] に加算されます。
    # これは異なるブロック内の点にアクセスすることを意味します。元のコードには境界チェックがありません。
    # arr の境界内で有効なインデックスにアクセスしていると仮定します。
    cdef int[120][3] combi = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 0, 6], [0, 0, 7], [0, 1, 1], [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 1, 6], [0, 1, 7], [0, 2, 2], [0, 2, 3], [0, 2, 4], [0, 2, 5], [0, 2, 6], [0, 2, 7], [0, 3, 3], [0, 3, 4], [0, 3, 5], [0, 3, 6], [0, 3, 7], [0, 4, 4], [0, 4, 5], [0, 4, 6], [0, 4, 7], [0, 5, 5], [0, 5, 6], [0, 5, 7], [0, 6, 6], [0, 6, 7], [0, 7, 7], [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4], [1, 1, 5], [1, 1, 6], [1, 1, 7], [1, 2, 2], [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6], [1, 2, 7], [1, 3, 3], [1, 3, 4], [1, 3, 5], [1, 3, 6], [1, 3, 7], [1, 4, 4], [1, 4, 5], [1, 4, 6], [1, 4, 7], [1, 5, 5], [1, 5, 6], [1, 5, 7], [1, 6, 6], [1, 6, 7], [1, 7, 7], [2, 2, 2], [2, 2, 3], [2, 2, 4], [2, 2, 5], [2, 2, 6], [2, 2, 7], [2, 3, 3], [2, 3, 4], [2, 3, 5], [2, 3, 6], [2, 3, 7], [2, 4, 4], [2, 4, 5], [2, 4, 6], [2, 4, 7], [2, 5, 5], [2, 5, 6], [2, 5, 7], [2, 6, 6], [2, 6, 7], [2, 7, 7], [3, 3, 3], [3, 3, 4], [3, 3, 5], [3, 3, 6], [3, 3, 7], [3, 4, 4], [3, 4, 5], [3, 4, 6], [3, 4, 7], [3, 5, 5], [3, 5, 6], [3, 5, 7], [3, 6, 6], [3, 6, 7], [3, 7, 7], [4, 4, 4], [4, 4, 5], [4, 4, 6], [4, 4, 7], [4, 5, 5], [4, 5, 6], [4, 5, 7], [4, 6, 6], [4, 6, 7], [4, 7, 7], [5, 5, 5], [5, 5, 6], [5, 5, 7], [5, 6, 6], [5, 6, 7], [5, 7, 7], [6, 6, 6], [6, 6, 7], [6, 7, 7], [7, 7, 7]]

    cdef int a, b, c, d, e
    cdef cnp.ndarray[cnp.int32_t, ndim=1] box_indices_for_permutation
    cdef cnp.ndarray[cnp.int32_t, ndim=2] codebook_for_permutation

    # apply_median_cut の目標箱数を決定。
    # 元のラッパーには指定がありませんでした。デフォルトとして、例えば256を使用します。
    # この値は必要に応じて変更できます。
    # 0x7FFFFFFF のような大きな値を使用すると、分散閾値に達するか箱のサイズが1になるまで事実上分割します。
    cdef int desired_num_boxes = 0x7FFFFFFF

    # combi で定義された各順列を繰り返し処理
    # 外側のループは120回実行され、毎回異なる tmp_in を準備し、メディアンカットを適用します。
    for a in range(120):
        # 現在の順列 'a' に対して tmp_in を埋める。
        # この部分は、元のラッパーと同様に prange を使用し、ループに nogil を適用しています。
        # インデックス計算ロジックは提供されたラッパーコードから直接取得しています。
        with nogil:
            for b in prange(num_blocks_x, schedule='static'):
                for c in prange(num_blocks_y):
                    for d in prange(num_blocks_z):
                        # 現在のブロック/点の tmp_in 内の線形インデックスを計算
                        e = b * num_blocks_y * num_blocks_z + c * num_blocks_z + d

                        # 現在の順列 combi[a] とブロックインデックス (b, c, d) に基づいて arr から6つの特徴を抽出。
                        # これは combi オフセットに基づいて異なるブロック内の点にアクセスします。
                        # インデックス b + combi[a][i], c + combi[a][j], d + combi[a][k] が arr の境界内にあると仮定します。
                        # 元のコードは tmp を使用しており、これは arr です。
                        # ここでの arr へのアクセスは、arr が C-contiguous であっても Cython が正しく処理します。
                        # tmp_in への書き込みは、tmp_in が F-contiguous であれば列方向に連続して行われます。
                        tmp_in[0, e] = arr[b + combi[a][0], c + combi[a][1], d + combi[a][2]]
                        tmp_in[1, e] = arr[b + combi[a][0], c + combi[a][2], d + combi[a][1]]
                        tmp_in[2, e] = arr[b + combi[a][1], c + combi[a][0], d + combi[a][2]]
                        tmp_in[3, e] = arr[b + combi[a][1], c + combi[a][2], d + combi[a][0]]
                        tmp_in[4, e] = arr[b + combi[a][2], c + combi[a][0], d + combi[a][1]]
                        tmp_in[5, e] = arr[b + combi[a][2], c + combi[a][1], d + combi[a][0]]

        # この順列のために準備されたデータ (tmp_in) にコアメディアンカットアルゴリズムを適用。
        # 各点の箱インデックスとコードブックを返します。
        # apply_median_cut は cpdef 関数であり、内部で独自のGIL管理を行います。
        box_indices_for_permutation, codebook_for_permutation = apply_median_cut(tmp_in, desired_num_boxes)

        # 返された箱インデックスを t1 出力配列の a 番目の行にコピー。
        # これには box_indices_for_permutation 配列を繰り返し処理する必要があります。
        # これはメモリビューにのみアクセスするため、nogil で実行可能。
        with nogil:
            for i in range(total_points):
                t1[a, i] = box_indices_for_permutation[i]

        # この順列のために生成されたコードブックをコードブックのリストに追加。
        # Pythonリストへの追加にはGILが必要です。
        cd_list.append(codebook_for_permutation)

    # 元のラッパーは t1 と cd_list を返します。
    # t1 は120個の順列それぞれの箱インデックスを含みます。
    # cd_list は120個のコードブックを含みます。
    return t1_arr, cd_list
