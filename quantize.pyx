# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
import cython
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrtf, fabsf
import numpy as np
cimport numpy as np

##########################################################
# 内部ヘルパー関数 (cdef inline) の定義
# これらはモジュールレベルで定義され、cpdef関数から呼び出されます。
##########################################################

cdef inline void compute_box_bounds(int* order, float* pts, int start, int end,
                                     float* min_x, float* max_x, float* min_y, float* max_y) nogil:
    """
    現在のボックスに含まれる点（order 配列で指定）について、x, y の最小／最大値を計算する。
    """
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
        if pts[idx_local*2] < min_x[0]:
            min_x[0] = pts[idx_local*2]
        elif pts[idx_local*2] > max_x[0]:
            max_x[0] = pts[idx_local*2]
        if pts[idx_local*2+1] < min_y[0]:
            min_y[0] = pts[idx_local*2+1]
        elif pts[idx_local*2+1] > max_y[0]:
            max_y[0] = pts[idx_local*2+1]

cdef inline int partition(int* order, float* pts, int left, int right, int dim) nogil:
    """
    与えられた区間 [left, right) に対して、dim 番目（0: x, 1: y）を用いたパーティショニングを行う。
    ピボットは区間最後尾の要素とし、order 配列を再配置する。
    """
    cdef int i = left, j, idx_local, tmp
    cdef float pivot_val = pts[ order[right-1] * 2 + dim ]
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
    """
    order 配列の区間 [left, right) について、k 番目の値が決まるように区画分割を繰り返す。
    """
    cdef int pivot_index
    while right - left > 1:
        pivot_index = partition(order, pts, left, right, dim)
        if pivot_index == k:
            return pivot_index
        elif pivot_index < k:
            left = pivot_index + 1
        else:
            right = pivot_index
    return left

cdef inline void compute_centroid(int* order, float* pts, int start, int end,
                                   float* centroid_x, float* centroid_y) nogil:
    """
    指定区間 [start, end) の点について、x, y の平均（重心）を計算する。
    """
    cdef float sumx = 0.0, sumy = 0.0
    cdef int k, idx_local, n = end - start
    if n == 0: # Handle empty box case
        centroid_x[0] = 0.0
        centroid_y[0] = 0.0
        return
    for k in range(start, end):
        idx_local = order[k]
        sumx += pts[idx_local*2]
        sumy += pts[idx_local*2+1]
    centroid_x[0] = sumx / n
    centroid_y[0] = sumy / n

cdef inline void median_cut_clustering_internal(float* pts, int num_pts, int n_clusters_target,
                                                 int* order, int* cluster_assign_temp, float* codebook_temp,
                                                 int* actual_n_clusters) nogil:
    """
    メディアンカットアルゴリズムにより、pts に含まれる点 (order で索引付け)
    を n_clusters_target 個になるまでボックスに分割し、各ボックスの重心を codebook_temp として返す。
    また、各点がどのボックスに所属しているかを cluster_assign_temp に 0 から順に割り当てる。
    実際のクラスタ数は actual_n_clusters に格納される。
    """
    cdef int max_boxes = n_clusters_target # 最大で目標クラスタ数までボックスを作成
    cdef int* box_start = <int*> malloc(max_boxes * sizeof(int))
    cdef int* box_end   = <int*> malloc(max_boxes * sizeof(int))
    cdef float* box_min_x = <float*> malloc(max_boxes * sizeof(float))
    cdef float* box_max_x = <float*> malloc(max_boxes * sizeof(float))
    cdef float* box_min_y = <float*> malloc(max_boxes * sizeof(float))
    cdef float* box_max_y = <float*> malloc(max_boxes * sizeof(float))

    if not box_start or not box_end or not box_min_x or not box_max_x or not box_min_y or not box_max_y:
        # メモリ確保失敗時はエラー処理が必要だが、nogil内なので単純にreturn
        # 呼び出し元でNULLチェックが必要
        actual_n_clusters[0] = 0
        return

    cdef int num_boxes = 1
    cdef int i, selected_box, mid, dim, n_in_box
    cdef float range_x, range_y, current_range, max_range
    cdef float centroid_x_val, centroid_y_val

    # 初期ボックスは全点を覆う
    box_start[0] = 0
    box_end[0] = num_pts
    compute_box_bounds(order, pts, 0, num_pts, &box_min_x[0], &box_max_x[0], &box_min_y[0], &box_max_y[0])
    cdef:
        int current_box_start, current_box_end
        float current_box_min_x, current_box_max_x, current_box_min_y, current_box_max_y
    # ボックスを n_clusters_target 個になるまで分割
    while num_boxes < n_clusters_target:
        max_range = -1.0
        selected_box = -1
        for i in range(num_boxes):
            n_in_box = box_end[i] - box_start[i]
            if n_in_box <= 1:
                continue # 点が1つ以下のボックスは分割しない
            range_x = box_max_x[i] - box_min_x[i]
            range_y = box_max_y[i] - box_min_y[i]
            # 分割軸の選択基準: 最も範囲の大きい軸
            current_range = range_x if range_x >= range_y else range_y
            if current_range > max_range:
                max_range = current_range
                selected_box = i

        if selected_box == -1:
            break  # これ以上の分割は不可能 (全てのボックスが点1つ以下になったなど)

        # 分割対象のボックスの情報を取得
        current_box_start = box_start[selected_box]
        current_box_end = box_end[selected_box]
        current_box_min_x = box_min_x[selected_box]
        current_box_max_x = box_max_x[selected_box]
        current_box_min_y = box_min_y[selected_box]
        current_box_max_y = box_max_y[selected_box]

        # 選択されたボックスをリストの最後に移動（削除＆追加の代わりに位置を入れ替える）
        # これにより、削除による要素の移動を避ける
        if selected_box != num_boxes - 1:
            box_start[selected_box] = box_start[num_boxes - 1]
            box_end[selected_box] = box_end[num_boxes - 1]
            box_min_x[selected_box] = box_min_x[num_boxes - 1]
            box_max_x[selected_box] = box_max_x[num_boxes - 1]
            box_min_y[selected_box] = box_min_y[num_boxes - 1]
            box_max_y[selected_box] = box_max_y[num_boxes - 1]

        # 分割軸を決定 (x 座標と y 座標のうち広がりが大きい方)
        range_x = current_box_max_x - current_box_min_x
        range_y = current_box_max_y - current_box_min_y
        if range_x >= range_y:
            dim = 0 # x軸で分割
        else:
            dim = 1 # y軸で分割

        # 中央値を見つけて分割
        # quickselect は order 配列をその場で並べ替える
        mid = (current_box_start + current_box_end) >> 1  # 中央位置のインデックス
        quickselect(order, pts, current_box_start, current_box_end, mid, dim)

        # 新しい2つのボックスの範囲を設定
        # 1つ目のボックス: [current_box_start, mid)
        box_start[num_boxes - 1] = current_box_start
        box_end[num_boxes - 1] = mid
        compute_box_bounds(order, pts, box_start[num_boxes - 1], box_end[num_boxes - 1],
                           &box_min_x[num_boxes - 1], &box_max_x[num_boxes - 1],
                           &box_min_y[num_boxes - 1], &box_max_y[num_boxes - 1])

        # 2つ目のボックス: [mid, current_box_end)
        box_start[num_boxes] = mid
        box_end[num_boxes] = current_box_end
        compute_box_bounds(order, pts, box_start[num_boxes], box_end[num_boxes],
                           &box_min_x[num_boxes], &box_max_x[num_boxes],
                           &box_min_y[num_boxes], &box_max_y[num_boxes])

        num_boxes += 1 # ボックス数をインクリメント

    # クラスタリング完了: 各ボックスの重心を計算して codebook_temp に格納
    # 各点がどのボックスに所属するかを cluster_assign_temp に割り当て
    actual_n_clusters[0] = num_boxes # 実際に作成されたボックス数

    for i in range(num_boxes):
        # ボックスの重心を計算
        compute_centroid(order, pts, box_start[i], box_end[i], &centroid_x_val, &centroid_y_val)
        codebook_temp[i*2] = centroid_x_val
        codebook_temp[i*2+1] = centroid_y_val

        # そのボックスに属する点のクラスタ番号を割り当て
        for j in range(box_start[i], box_end[i]):
            cluster_assign_temp[order[j]] = i # 0-based index

    # メモリ解放
    free(box_start)
    free(box_end)
    free(box_min_x)
    free(box_max_x)
    free(box_min_y)
    free(box_max_y)


cdef inline float euclidean_distance_pts(float* p1, float* p2) nogil:
    """
    2つの点 (x, y) 間のユークリッド距離を計算します。
    コンパイラが /arch:SSE4.2 オプションで最適化を行うことを期待します。
    """
    cdef float dx = p1[0] - p2[0]
    cdef float dy = p1[1] - p2[1]
    return sqrtf(dx*dx + dy*dy)


# Python側から直接アクセスするため、cpdef としています。
cpdef quantize_data(float[:, :, :] data, int max_clusters_to_test):
    cdef int H = data.shape[0]
    cdef int W = data.shape[1]
    cdef int total = H * W
    cdef int num_non_zero_points = 0
    cdef int i, j, k

    if max_clusters_to_test < 1:
         raise ValueError("max_clusters_to_testは1以上の整数である必要があります。")

    # まず、(x, y) が (0, 0) である点はクラスタリング処理対象外とする
    for i in range(H):
        for j in range(W):
            if not (data[i,j,0] == 0.0 and data[i,j,1] == 0.0):
                num_non_zero_points += 1

    # 非ゼロ点が一つもない場合は空の結果を返す
    if num_non_zero_points == 0:
        print("Info: 全ての点が (0, 0) です。")
        out_indices = np.zeros((H, W, 1), dtype=np.intc)
        out_codebook = np.empty((0,2), dtype=np.float32)
        return out_codebook, out_indices

    # 非ゼロの点を C 配列にコピーし、元の画像上の位置も保持する
    cdef float* points = <float*> malloc(num_non_zero_points * 2 * sizeof(float))
    cdef int* orig_i = <int*> malloc(num_non_zero_points * sizeof(int))
    cdef int* orig_j = <int*> malloc(num_non_zero_points * sizeof(int))
    # メディアンカット用の点の順序を保持する配列 (初期状態では 0 から num_non_zero_points-1)
    cdef int* order = <int*> malloc(num_non_zero_points * sizeof(int))

    if not points or not orig_i or not orig_j or not order:
        # 確保済みのメモリがあれば解放
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
            order[idx] = idx # order配列を初期化
            idx += 1

    # エルボー法のためのMAEを格納するリスト
    cdef list mae_values = []

    # エルボー法のためのクラスタ数ループ (1からmax_clusters_to_testまで)
    # 各クラスタ数 k について、メディアンカットを実行し MAE を計算する
    cdef int n_clusters_k
    cdef float total_mae
    cdef float point_mae
    cdef int actual_n_clusters_k # 実際に作成されたクラスタ数
    cdef int assigned_cluster_idx # 0-based index

    # 各クラスタ数 k でのクラスタ割り当て結果を一時的に保持する配列
    cdef int* cluster_assign_temp = <int*> malloc(num_non_zero_points * sizeof(int))
    # 各クラスタ数 k でのコードブックを一時的に保持する配列 (最大 max_clusters_to_test * 2)
    cdef float* codebook_temp = <float*> malloc(max_clusters_to_test * 2 * sizeof(float))

    if not cluster_assign_temp or not codebook_temp:
        # 確保済みのメモリを全て解放
        if points: free(points)
        if orig_i: free(orig_i)
        if orig_j: free(orig_j)
        if order: free(order)
        if cluster_assign_temp: free(cluster_assign_temp)
        if codebook_temp: free(codebook_temp)
        raise MemoryError("一時配列のメモリ確保に失敗しました。")
    cdef float centroid_x, centroid_y, average_mae
    for n_clusters_k in range(1, max_clusters_to_test + 1):
        # メディアンカットクラスタリングを実行 (一時配列を使用)
        # order配列は median_cut_clustering_internal 内で変更されるため、毎回初期状態に戻す必要がある
        for i in range(num_non_zero_points):
             order[i] = i

        median_cut_clustering_internal(points, num_non_zero_points, n_clusters_k,
                                       order, cluster_assign_temp, codebook_temp,
                                       &actual_n_clusters_k)

        # 実際に作成されたクラスタ数が0の場合はスキップ (通常は発生しないはずだが安全策)
        if actual_n_clusters_k == 0:
             mae_values.append(0.0) # MAEを0として扱うか、適切な値を検討
             continue

        # 平均絶対誤差 (MAE) を計算
        total_mae = 0.0
        # prangeを使って並列化
        for i in prange(num_non_zero_points, nogil=True):
            assigned_cluster_idx = cluster_assign_temp[i] # 0-based index
            # 所属するクラスタの重心座標
            centroid_x = codebook_temp[assigned_cluster_idx*2]
            centroid_y = codebook_temp[assigned_cluster_idx*2+1]

            # 元の点と重心とのユークリッド距離
            point_mae = euclidean_distance_pts(&points[i*2], &codebook_temp[assigned_cluster_idx*2])
            total_mae += point_mae

        average_mae = total_mae / num_non_zero_points
        mae_values.append(average_mae)
        # print(f"Info: MAE for n_clusters = {n_clusters_k} is {average_mae}") # デバッグ出力

    cdef int best_n_clusters = 1 # デフォルトは1クラスタ
    cdef float max_distance_from_line = -1.0
    cdef float x1, y1, x2, y2, m, c, current_n_float, current_mae, distance
    cdef int n_clusters_eval
    if max_clusters_to_test > 1:
        # MAE値は n=1 から max_clusters_to_test までのもの
        # リストのインデックスは 0 から max_clusters_to_test - 1
        x1 = 1.0
        y1 = mae_values[0] # MAE for n=1
        x2 = <float>max_clusters_to_test
        y2 = mae_values[max_clusters_to_test - 1] # MAE for n=max_clusters_to_test

        # 直線の式: y = mx + c
        # m = (y2 - y1) / (x2 - x1)
        # c = y1 - m * x1
        # x1 == x2 の場合は垂直線になるが、max_clusters_to_test > 1 なので発生しない
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1

        # 各点 (n, MAE_n) から直線までの距離を計算

        # n=2 から max_clusters_to_test までを評価
        for n_clusters_eval in range(2, max_clusters_to_test + 1):
            current_n_float = <float>n_clusters_eval
            current_mae = mae_values[n_clusters_eval - 1] # MAE for current n (list index n-1)

            # 点 (x0, y0) から直線 Ax + By + C = 0 までの距離の公式: |Ax0 + By0 + C| / sqrt(A^2 + B^2)
            # 直線の式 y = mx + c を mx - y + c = 0 に変換
            # A = m, B = -1, C = c
            # x0 = current_n_float, y0 = current_mae
            distance = fabsf(m * current_n_float - current_mae + c) / sqrtf(m*m + (-1.0)*(-1.0))

            if distance > max_distance_from_line:
                max_distance_from_line = distance
                best_n_clusters = n_clusters_eval # この n が最適なクラスタ数

    # print(f"Info: Optimal number of clusters determined by Elbow method is {best_n_clusters}") # デバッグ出力

    # 最適なクラスタ数で再度メディアンカットと量子化を実行
    # order配列を再度初期化
    for i in range(num_non_zero_points):
         order[i] = i

    # 最終結果用のコードブックのメモリを malloc で確保
    # コードブックサイズは最適なクラスタ数 × 2 (x, y)
    cdef float* final_codebook_ptr = <float*> malloc(best_n_clusters * 2 * sizeof(float))

    # 量子化後のインデックス配列 (元の画像サイズ) は numpy で作成し、メモリビューを使用
    cdef int[:, :, :] final_indices_view = np.zeros((H, W, 1), dtype=np.intc)
    # final_indices_ptr は不要なので削除

    # 最終的なクラスタ割り当て結果を格納する配列
    cdef int* final_cluster_assign = <int*> malloc(num_non_zero_points * sizeof(int))

    if not final_codebook_ptr or not final_cluster_assign:
        # 確保済みのメモリを全て解放
        if points: free(points)
        if orig_i: free(orig_i)
        if orig_j: free(orig_j)
        if order: free(order)
        if cluster_assign_temp: free(cluster_assign_temp)
        if codebook_temp: free(codebook_temp)
        if final_codebook_ptr: free(final_codebook_ptr) # Free if allocated
        if final_cluster_assign: free(final_cluster_assign) # Free if allocated
        raise MemoryError("最終結果用配列のメモリ確保に失敗しました。")

    cdef int final_actual_n_clusters
    # median_cut_clustering_internal に malloc で確保したポインタを渡す
    median_cut_clustering_internal(points, num_non_zero_points, best_n_clusters,
                                   order, final_cluster_assign, final_codebook_ptr,
                                   &final_actual_n_clusters)

    # 量子化後のインデックス配列を生成
    # prangeを使って並列化
    for i in prange(num_non_zero_points, nogil=True):
        # 元の画像での位置に、所属するクラスタのインデックスを格納
        # クラスタインデックスは 0-based で返されるが、出力は 1-based とする
        final_indices_view[orig_i[i], orig_j[i], 0] = final_cluster_assign[i] + 1

    # malloc で確保したコードブックのメモリから numpy 配列を作成
    # 生のポインタからサイズを指定したメモリビューを作成し、それを np.asarray に渡す
    cdef float[:] codebook_memoryview = <float[:final_actual_n_clusters*2]>final_codebook_ptr
    cdef np.ndarray final_codebook_np = np.asarray(codebook_memoryview).reshape((final_actual_n_clusters, 2))

    # メモリ解放
    free(points)
    free(orig_i)
    free(orig_j)
    free(order)
    free(cluster_assign_temp)
    free(codebook_temp)
    free(final_cluster_assign)
    # final_codebook_ptr のメモリは numpy が管理するようになったため、ここでは free しない

    # 結果を返す
    # コードブックは numpy 配列として返す
    # final_actual_n_clusters に基づいてコードブックのサイズを調整して返す
    return final_codebook_np, final_indices_view # final_actual_n_clusters でサイズ調整済みなのでスライス不要
