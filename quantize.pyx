import math
import random

def quicksort(lst, key_func):
    """シンプルなクイックソートの実装（sorted() は使わず自前で）"""
    if len(lst) <= 1:
        return lst
    pivot = lst[len(lst) // 2]
    pivot_val = key_func(pivot)
    left, middle, right = [], [], []
    for x in lst:
        k = key_func(x)
        if k < pivot_val:
            left.append(x)
        elif k > pivot_val:
            right.append(x)
        else:
            middle.append(x)
    return quicksort(left, key_func) + middle + quicksort(right, key_func)

def compute_mean_and_distortion(points, indices):
    """
    箱内の平均値と歪み（各点の平均からの二乗距離の和）を計算する
    points: 2要素リスト（points[0]が x、points[1]が y）
    indices: 箱に所属する点のインデックスのリスト
    """
    n = len(indices)
    if n == 0:
        return ((0.0, 0.0), 0.0)
    sum_x = 0.0
    sum_y = 0.0
    for i in indices:
        sum_x += points[0][i]
        sum_y += points[1][i]
    mean_x = sum_x / n
    mean_y = sum_y / n
    distortion = 0.0
    for i in indices:
        dx = points[0][i] - mean_x
        dy = points[1][i] - mean_y
        distortion += dx * dx + dy * dy
    return ((mean_x, mean_y), distortion)

def median_cut(points, SPLIT_PENALTY=1.0, BIT_PENALTY=10.0):
    """
    メディアンカットアルゴリズムによる歪みレート最適化。
    
    入力:
      points: 2要素のシーケンス（例：[[x0, x1, …, x_(N-1)], [y0, y1, …, y_(N-1)]])
      SPLIT_PENALTY: 各分割ごとにかかる定数ペナルティ
      BIT_PENALTY: 箱の総数が2の冪の閾値を超えた場合にかかる大きなペナルティ
      
    戻り値:
      representatives: 各箱の代表値（箱内の平均値）のリスト [(x, y), …]
      labels: 各点が所属する箱のインデックスリスト（長さ N）
    """
    N = len(points[0])
    # 初期状態：全点を 1 つの箱にまとめる
    boxes = [{
        'indices': list(range(N)),
        'mean': None,
        'distortion': None
    }]
    mean, distortion = compute_mean_and_distortion(points, boxes[0]['indices'])
    boxes[0]['mean'] = mean
    boxes[0]['distortion'] = distortion
    current_total_clusters = 1  # 現在の箱数

    # どの箱も分割できなくなるまで、または改善が見込めなくなるまでループ
    while True:
        best_margin = -1e9
        best_box_index = None
        best_split = None

        # すべての箱について分割候補を評価する
        for i, box in enumerate(boxes):
            indices = box['indices']
            if len(indices) < 2:
                continue  # 1点以下の箱は分割できない
            # 箱内の x, y の最小・最大を求め、どの軸で分割するか決定
            xs = [points[0][j] for j in indices]
            ys = [points[1][j] for j in indices]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            range_x = max_x - min_x
            range_y = max_y - min_y
            # より広い軸を分割軸とする
            axis = 0 if range_x >= range_y else 1
            # クイックソートで指定軸によるソート
            sorted_indices = quicksort(indices, key_func=lambda j: points[axis][j])
            mid = len(sorted_indices) // 2
            if mid == 0 or mid == len(sorted_indices):
                continue  # 片側が空になってしまう場合はスキップ
            left_indices = sorted_indices[:mid]
            right_indices = sorted_indices[mid:]
            left_mean, left_dist = compute_mean_and_distortion(points, left_indices)
            right_mean, right_dist = compute_mean_and_distortion(points, right_indices)
            new_total_distortion = left_dist + right_dist
            improvement = box['distortion'] - new_total_distortion

            # グローバルな箱数が増える時，必ずしも箱番号の最低ビット数は一定とはならない。
            # ここでは current_total_clusters から箱が 1 増えるときに，
            # もしその数が (1→2, 3→4, …) のように2の冪の閾値を超えるなら BIT_PENALTY を加える
            old_bits = math.floor(math.log2(current_total_clusters)) + 1 if current_total_clusters > 0 else 1
            new_bits = math.floor(math.log2(current_total_clusters + 1)) + 1
            bit_penalty = BIT_PENALTY if new_bits > old_bits else 0
            cost_penalty = SPLIT_PENALTY + bit_penalty

            margin = improvement - cost_penalty
            # もっとも改善余地が大きい分割候補を記録
            if margin > best_margin:
                best_margin = margin
                best_box_index = i
                best_split = {
                    'left_indices': left_indices,
                    'right_indices': right_indices,
                    'left_mean': left_mean,
                    'left_dist': left_dist,
                    'right_mean': right_mean,
                    'right_dist': right_dist
                }
        # どこも改善できなければループ終了
        if best_box_index is None or best_margin <= 0:
            break

        # 選ばれた箱を分割する（分割候補の改善がペナルティを上回れば実行）
        del boxes[best_box_index]
        boxes.append({
            'indices': best_split['left_indices'],
            'mean': best_split['left_mean'],
            'distortion': best_split['left_dist']
        })
        boxes.append({
            'indices': best_split['right_indices'],
            'mean': best_split['right_mean'],
            'distortion': best_split['right_dist']
        })
        # 1つの箱が2つになったので、全体の箱数は +1 となる
        current_total_clusters += 1

    # 最終的な箱ごとに、各点の所属（label）を決定する
    labels = [None] * N
    for idx, box in enumerate(boxes):
        for j in box['indices']:
            labels[j] = idx

    # 箱の代表値（平均値）のリストを作成
    representatives = [box['mean'] for box in boxes]

    return representatives, labels

# --- 以下、動作確認のためのテスト例 ---
if __name__ == '__main__':
    # 例：ランダムに 10 個の点 (x, y) を生成（numpy は使わず random モジュールにより）
    N = 10
    x_vals = [random.uniform(0, 100) for _ in range(N)]
    y_vals = [random.uniform(0, 100) for _ in range(N)]
    points = [x_vals, y_vals]  # shape=(2, N)
    
    # メディアンカットによる最適化（ペナルティパラメータは好みに応じて調整可能）
    reps, labels = median_cut(points, SPLIT_PENALTY=1.0, BIT_PENALTY=10.0)
    
    print("各箱の代表値（平均）:")
    for i, rep in enumerate(reps):
        print(f"Box {i}: {rep}")
    print("\n各点の所属箱インデックス:")
    for i in range(N):
        print(f"Point {i} (x={points[0][i]:.2f}, y={points[1][i]:.2f}) → Box {labels[i]}")
