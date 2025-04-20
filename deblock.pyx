# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=False

import numpy as np
from cython.parallel import prange
from libc.math cimport fabs, fmax, fmin

# HVS適応型デブロッキングフィルター
# 8x8ブロックに対する適応型フィルターを実装します。
# HVS高速近似モデルに基づき、タップ数、フィルター強度、alpha, betaを動的に決定します。
# Y, U, Vコンポーネントそれぞれに異なる近似モデルを適用します。
# 入出力は unsigned short[:, :, :] のメモリビューを使用します。

# 定数定義
cdef int BLOCK_SIZE = 8
cdef int BORDER_SIZE = 4

# HVSモデルのパラメータ（調整可能）
# これらの閾値や係数は、実際のHVS特性に合わせて調整することで、
# フィルターの振る舞いを変更できます。
# Yコンポーネント用
cdef float Y_BOUNDARY_DIFF_THRESHOLD_FLAT = 8.0  # 境界差がこれ以下ならフラットとみなす閾値
cdef float Y_BOUNDARY_DIFF_THRESHOLD_EDGE = 32.0 # 境界差がこれ以上ならエッジとみなす閾値
cdef float Y_TEXTURE_THRESHOLD_FLAT = 16.0       # テクスチャがこれ以下ならフラットとみなす閾値
cdef float Y_TEXTURE_THRESHOLD_TEXTURED = 64.0   # テクスチャがこれ以上ならテクスチャとみなす閾値
cdef float Y_MAX_FILTER_STRENGTH = 64.0          # Yコンポーネントの最大フィルター強度
cdef float Y_ALPHA_FLAT = 0.1                    # フラット領域でのalpha初期値
cdef float Y_ALPHA_TEXTURED = 0.5                # テクスチャ領域でのalpha初期値
cdef float Y_BETA_FLAT = 0.5                     # フラット領域でのbeta初期値
cdef float Y_BETA_TEXTURED = 0.1                 # テクスチャ領域でのbeta初期値

# UVコンポーネント用（Yより感度が低いことを考慮）
cdef float UV_BOUNDARY_DIFF_THRESHOLD_FLAT = 16.0  # 境界差がこれ以下ならフラットとみなす閾値
cdef float UV_BOUNDARY_DIFF_THRESHOLD_EDGE = 64.0 # 境界差がこれ以上ならエッジとみなす閾値
cdef float UV_TEXTURE_THRESHOLD_FLAT = 32.0       # テクスチャがこれ以下ならフラットとみなす閾値
cdef float UV_TEXTURE_THRESHOLD_TEXTURED = 128.0  # テクスチャがこれ以上ならテクスチャとみなす閾値
cdef float UV_MAX_FILTER_STRENGTH = 32.0         # UVコンポーネントの最大フィルター強度
cdef float UV_ALPHA_FLAT = 0.2                   # フラット領域でのalpha初期値
cdef float UV_ALPHA_TEXTURED = 0.6               # テクスチャ領域でのalpha初期値
cdef float UV_BETA_FLAT = 0.6                    # フラット領域でのbeta初期値
cdef float UV_BETA_TEXTURED = 0.2                # テクスチャ領域でのbeta初期値

# HVSモデルに基づき、境界差とテクスチャからパラメータを滑らかに決定するヘルパー関数群

# 境界差とテクスチャからタップ数を決定
cdef inline int determine_tap_count(float boundary_diff, float texture, int plane_idx) nogil:
    cdef float boundary_flat_thresh, boundary_edge_thresh
    cdef float texture_flat_thresh, texture_textured_thresh

    # コンポーネントに応じて閾値を設定
    if plane_idx == 0: # Y
        boundary_flat_thresh = Y_BOUNDARY_DIFF_THRESHOLD_FLAT
        boundary_edge_thresh = Y_BOUNDARY_DIFF_THRESHOLD_EDGE
        texture_flat_thresh = Y_TEXTURE_THRESHOLD_FLAT
        texture_textured_thresh = Y_TEXTURE_THRESHOLD_TEXTURED
    else: # UV
        boundary_flat_thresh = UV_BOUNDARY_DIFF_THRESHOLD_FLAT
        boundary_edge_thresh = UV_BOUNDARY_DIFF_THRESHOLD_EDGE
        texture_flat_thresh = UV_TEXTURE_THRESHOLD_FLAT
        texture_textured_thresh = UV_TEXTURE_THRESHOLD_TEXTURED

    # 境界差が大きい場合はエッジとみなし、タップ数を減らす
    if boundary_diff > boundary_edge_thresh:
        return 0 # エッジが強い場合はフィルターしない
    elif boundary_diff > boundary_flat_thresh:
        # 境界差が中程度の場合は、境界差に応じてタップ数を線形補間
        # 境界差が大きいほどタップ数を減らす
        cdef float ratio = (boundary_diff - boundary_flat_thresh) / (boundary_edge_thresh - boundary_flat_thresh)
        # テクスチャも考慮してタップ数を調整
        cdef float texture_ratio = (texture - texture_flat_thresh) / (texture_textured_thresh - texture_flat_thresh)
        texture_ratio = fmax(0.0, fmin(1.0, texture_ratio)) # 0から1にクランプ
        # テクスチャが多いほどタップ数を減らす
        cdef float tap_ratio = fmin(ratio + texture_ratio, 1.0) # 境界差とテクスチャの両方でタップ数を減らす方向へ
        if tap_ratio < 0.2: return 8
        elif tap_ratio < 0.4: return 6
        elif tap_ratio < 0.6: return 4
        elif tap_ratio < 0.8: return 2
        else: return 0
    else:
        # 境界差が小さい場合はフラット領域とみなし、テクスチャに応じてタップ数を決定
        if texture < texture_flat_thresh:
            return 8 # 非常にフラットなら最大タップ
        elif texture < texture_textured_thresh:
            # テクスチャが中程度の場合は、テクスチャに応じてタップ数を線形補間
            cdef float ratio = (texture - texture_flat_thresh) / (texture_textured_thresh - texture_flat_thresh)
            if ratio < 0.25: return 8
            elif ratio < 0.5: return 6
            elif ratio < 0.75: return 4
            else: return 2
        else:
            return 0 # テクスチャが多い場合はフィルターしない

# 境界差とテクスチャからフィルター強度を決定
cdef inline float determine_filter_strength(float boundary_diff, float texture, int plane_idx) nogil:
    cdef float boundary_flat_thresh, boundary_edge_thresh
    cdef float texture_flat_thresh, texture_textured_thresh
    cdef float max_strength

    # コンポーネントに応じて閾値を設定
    if plane_idx == 0: # Y
        boundary_flat_thresh = Y_BOUNDARY_DIFF_THRESHOLD_FLAT
        boundary_edge_thresh = Y_BOUNDARY_DIFF_THRESHOLD_EDGE
        texture_flat_thresh = Y_TEXTURE_THRESHOLD_FLAT
        texture_textured_thresh = Y_TEXTURE_THRESHOLD_TEXTURED
        max_strength = Y_MAX_FILTER_STRENGTH
    else: # UV
        boundary_flat_thresh = UV_BOUNDARY_DIFF_THRESHOLD_FLAT
        boundary_edge_thresh = UV_BOUNDARY_DIFF_THRESHOLD_EDGE
        texture_flat_thresh = UV_TEXTURE_THRESHOLD_FLAT
        texture_textured_thresh = UV_TEXTURE_THRESHOLD_TEXTURED
        max_strength = UV_MAX_FILTER_STRENGTH

    # 境界差とテクスチャが小さいほど強度を高くする
    cdef float boundary_ratio = 1.0 - fmax(0.0, fmin(1.0, (boundary_diff - boundary_flat_thresh) / (boundary_edge_thresh - boundary_flat_thresh)))
    cdef float texture_ratio = 1.0 - fmax(0.0, fmin(1.0, (texture - texture_flat_thresh) / (texture_textured_thresh - texture_textured_thresh)))

    # 境界差とテクスチャの両方を考慮して強度を決定（例：平均）
    cdef float strength_ratio = (boundary_ratio + texture_ratio) / 2.0

    return max_strength * strength_ratio

# 境界差とテクスチャからalphaを決定
cdef inline float determine_alpha(float boundary_diff, float texture, float strength, int plane_idx) nogil:
    cdef float boundary_flat_thresh, boundary_edge_thresh
    cdef float texture_flat_thresh, texture_textured_thresh
    cdef float alpha_flat, alpha_textured

    # コンポーネントに応じて閾値を設定
    if plane_idx == 0: # Y
        boundary_flat_thresh = Y_BOUNDARY_DIFF_THRESHOLD_FLAT
        boundary_edge_thresh = Y_BOUNDARY_DIFF_THRESHOLD_EDGE
        texture_flat_thresh = Y_TEXTURE_THRESHOLD_FLAT
        texture_textured_thresh = Y_TEXTURE_THRESHOLD_TEXTURED
        alpha_flat = Y_ALPHA_FLAT
        alpha_textured = Y_ALPHA_TEXTURED
    else: # UV
        boundary_flat_thresh = UV_BOUNDARY_DIFF_THRESHOLD_FLAT
        boundary_edge_thresh = UV_BOUNDARY_DIFF_THRESHOLD_EDGE
        texture_flat_thresh = UV_TEXTURE_THRESHOLD_FLAT
        texture_textured_thresh = UV_TEXTURE_THRESHOLD_TEXTURED
        alpha_flat = UV_ALPHA_FLAT
        alpha_textured = UV_ALPHA_TEXTURED

    # テクスチャが多いほどalphaを大きくする
    cdef float texture_ratio = fmax(0.0, fmin(1.0, (texture - texture_flat_thresh) / (texture_textured_thresh - texture_flat_thresh)))
    cdef float alpha = alpha_flat + (alpha_textured - alpha_flat) * texture_ratio

    # 強度も考慮してalphaを調整（例：強度が低いほどalphaも小さくする）
    cdef float max_strength = Y_MAX_FILTER_STRENGTH if plane_idx == 0 else UV_MAX_FILTER_STRENGTH
    alpha *= (strength / max_strength) if max_strength > 0 else 0.0

    return fmax(0.0, fmin(1.0, alpha)) # 0から1にクランプ

# 境界差とテクスチャからbetaを決定
cdef inline float determine_beta(float boundary_diff, float texture, float strength, int plane_idx) nogil:
    cdef float boundary_flat_thresh, boundary_edge_thresh
    cdef float texture_flat_thresh, texture_textured_thresh
    cdef float beta_flat, beta_textured

    # コンポーネントに応じて閾値を設定
    if plane_idx == 0: # Y
        boundary_flat_thresh = Y_BOUNDARY_DIFF_THRESHOLD_FLAT
        boundary_edge_thresh = Y_BOUNDARY_DIFF_THRESHOLD_EDGE
        texture_flat_thresh = Y_TEXTURE_THRESHOLD_FLAT
        texture_textured_thresh = Y_TEXTURE_THRESHOLD_TEXTURED
        beta_flat = Y_BETA_FLAT
        beta_textured = Y_BETA_TEXTURED
    else: # UV
        boundary_flat_thresh = UV_BOUNDARY_DIFF_THRESHOLD_FLAT
        boundary_edge_thresh = UV_BOUNDARY_DIFF_THRESHOLD_EDGE
        texture_flat_thresh = UV_TEXTURE_THRESHOLD_FLAT
        texture_textured_thresh = UV_TEXTURE_THRESHOLD_TEXTURED
        beta_flat = UV_BETA_FLAT
        beta_textured = UV_BETA_TEXTURED

    # テクスチャが多いほどbetaを小さくする
    cdef float texture_ratio = fmax(0.0, fmin(1.0, (texture - texture_flat_thresh) / (texture_textured_thresh - texture_flat_thresh)))
    cdef float beta = beta_flat + (beta_textured - beta_flat) * (1.0 - texture_ratio)

    # 強度も考慮してbetaを調整（例：強度が低いほどbetaも小さくする）
    cdef float max_strength = Y_MAX_FILTER_STRENGTH if plane_idx == 0 else UV_MAX_FILTER_STRENGTH
    beta *= (strength / max_strength) if max_strength > 0 else 0.0

    return fmax(0.0, fmin(1.0, beta)) # 0から1にクランプ

# HVSモデルに使用するメトリクス計算ヘルパー関数

# 境界を挟んだ画素値の差の絶対値の合計を計算
cdef inline float calculate_boundary_diff(unsigned short[:, :, :] image, int r, int c, int plane, bint is_horizontal) nogil:
    cdef int height = image.shape[0]
    cdef int width = image.shape[1]
    cdef float diff = 0.0
    cdef int i

    if is_horizontal: # 水平方向の境界 (行 r, 列 c と c-1 の間)
        if c == 0: return 0.0 # 画像の左端
        # 境界を挟んだ数ピクセルの差を合計
        for i in range(-2, 3): # 境界から上下2ピクセルずつ
            if r + i >= 0 and r + i < height:
                diff += fabs(image[r + i, c, plane] - image[r + i, c - 1, plane])
    else: # 垂直方向の境界 (行 r と r-1 の間, 列 c)
        if r == 0: return 0.0 # 画像の上端
        # 境界を挟んだ数ピクセルの差を合計
        for i in range(-2, 3): # 境界から左右2ピクセルずつ
            if c + i >= 0 and c + i < width:
                diff += fabs(image[r, c + i, plane] - image[r - 1, c + i, plane])

    return diff

# 境界付近のテクスチャ（分散の近似として差分の絶対値の合計を使用）を計算
cdef inline float calculate_local_texture(unsigned short[:, :, :] image, int r, int c, int plane, bint is_horizontal) nogil:
    cdef int height = image.shape[0]
    cdef int width = image.shape[1]
    cdef float texture = 0.0
    cdef int i, j
    cdef int window_size = 5 # 境界付近のテクスチャを評価するウィンドウサイズ (例: 5x5)

    if is_horizontal: # 水平方向の境界 (行 r, 列 c と c-1 の間)
        if c == 0: return 0.0
        # 境界を挟んだウィンドウ内の差分絶対値の合計
        for i in range(-window_size // 2, window_size // 2 + 1):
            for j in range(-window_size // 2, window_size // 2 + 1):
                 if r + i >= 0 and r + i < height and c + j >= 0 and c + j < width - 1:
                     texture += fabs(image[r + i, c + j + 1, plane] - image[r + i, c + j, plane])
    else: # 垂直方向の境界 (行 r と r-1 の間, 列 c)
        if r == 0: return 0.0
        # 境界を挟んだウィンドウ内の差分絶対値の合計
        for i in range(-window_size // 2, window_size // 2 + 1):
            for j in range(-window_size // 2, window_size // 2 + 1):
                 if r + i >= 0 and r + i < height - 1 and c + j >= 0 and c + j < width:
                     texture += fabs(image[r + i + 1, c + j, plane] - image[r + i, c + j, plane])

    return texture

# フィルター適用関数群 (タップ数ごと)
# フィルターの重みは、強度、alpha, betaに基づいて計算されます。

# 2タップフィルター
cdef inline void apply_filter_2tap(unsigned short[:, :, :] image, int r, int c, int plane, bint is_horizontal, float strength, float alpha, float beta) nogil:
    # 2タップフィルターは境界を挟んだ2画素に適用
    # 例: 水平方向の場合、image[r, c-1, plane] と image[r, c, plane] を平滑化
    cdef int p0, q0
    cdef int delta
    cdef int filtered_p0, filtered_q0

    if is_horizontal:
        p0 = image[r, c - 1, plane]
        q0 = image[r, c, plane]
    else: # 垂直方向
        p0 = image[r - 1, c, plane]
        q0 = image[r, c, plane]

    delta = <int>(round((q0 - p0) * alpha)) # alphaに基づいて差分を調整
    delta = <int>(fmax(-strength, fmin(strength, delta))) # 強度でクリップ

    filtered_p0 = p0 + delta
    filtered_q0 = q0 - delta

    # 結果を元の画素値の範囲にクランプ
    filtered_p0 = <int>(fmax(0.0, fmin(65535.0, filtered_p0)))
    filtered_q0 = <int>(fmax(0.0, fmin(65535.0, filtered_q0)))

    if is_horizontal:
        image[r, c - 1, plane] = filtered_p0
        image[r, c, plane] = filtered_q0
    else:
        image[r - 1, c, plane] = filtered_p0
        image[r, c, plane] = filtered_q0

# 4タップフィルター
cdef inline void apply_filter_4tap(unsigned short[:, :, :] image, int r, int c, int plane, bint is_horizontal, float strength, float alpha, float beta) nogil:
    # 4タップフィルターは境界を挟んだ4画素に適用
    # 例: 水平方向の場合、image[r, c-2, plane], image[r, c-1, plane], image[r, c, plane], image[r, c+1, plane]
    cdef int p1, p0, q0, q1
    cdef int delta
    cdef int filtered_p1, filtered_p0, filtered_q0, filtered_q1

    if is_horizontal:
        if c < 2 or c >= image.shape[1] - 1: return # 境界の端では適用しない
        p1 = image[r, c - 2, plane]
        p0 = image[r, c - 1, plane]
        q0 = image[r, c, plane]
        q1 = image[r, c + 1, plane]
    else: # 垂直方向
        if r < 2 or r >= image.shape[0] - 1: return # 境界の端では適用しない
        p1 = image[r - 2, c, plane]
        p0 = image[r - 1, c, plane]
        q0 = image[r, c, plane]
        q1 = image[r + 1, c, plane]
    # シンプルな4タップフィルターの例（重みは調整可能）
    # ここでは、境界差とbetaに基づいて平滑化の度合いを調整
    delta = <int>(round((q0 - p0) * alpha)) # alphaに基づいて差分を調整
    delta = <int>(fmax(-strength, fmin(strength, delta))) # 強度でクリップ

    # betaを使用して、境界から離れた画素への影響を調整
    cdef float beta_factor = 1.0 - beta # betaが大きいほど、境界から離れた画素への影響が小さくなる

    filtered_p0 = p0 + delta
    filtered_q0 = q0 - delta
    filtered_p1 = p1 + <int>(round(delta * beta_factor)) # beta_factorで調整
    filtered_q1 = q1 - <int>(round(delta * beta_factor)) # beta_factorで調整

    # 結果を元の画素値の範囲にクランプ
    filtered_p1 = <int>(fmax(0.0, fmin(65535.0, filtered_p1)))
    filtered_p0 = <int>(fmax(0.0, fmin(65535.0, filtered_p0)))
    filtered_q0 = <int>(fmax(0.0, fmin(65535.0, filtered_q0)))
    filtered_q1 = <int>(fmax(0.0, fmin(65535.0, filtered_q1)))


    if is_horizontal:
        image[r, c - 2, plane] = filtered_p1
        image[r, c - 1, plane] = filtered_p0
        image[r, c, plane] = filtered_q0
        image[r, c + 1, plane] = filtered_q1
    else:
        image[r - 2, c, plane] = filtered_p1
        image[r - 1, c, plane] = filtered_p0
        image[r, c, plane] = filtered_q0
        image[r + 1, c, plane] = filtered_q1


# 6タップフィルター
cdef inline void apply_filter_6tap(unsigned short[:, :, :] image, int r, int c, int plane, bint is_horizontal, float strength, float alpha, float beta) nogil:
    # 6タップフィルターは境界を挟んだ6画素に適用
    # 例: 水平方向の場合、image[r, c-3] ～ image[r, c+2]
    cdef int p2, p1, p0, q0, q1, q2
    cdef int delta
    cdef int filtered_p2, filtered_p1, filtered_p0, filtered_q0, filtered_q1, filtered_q2

    if is_horizontal:
        if c < 3 or c >= image.shape[1] - 2: return # 境界の端では適用しない
        p2 = image[r, c - 3, plane]
        p1 = image[r, c - 2, plane]
        p0 = image[r, c - 1, plane]
        q0 = image[r, c, plane]
        q1 = image[r, c + 1, plane]
        q2 = image[r, c + 2, plane]
    else: # 垂直方向
        if r < 3 or r >= image.shape[0] - 2: return # 境界の端では適用しない
        p2 = image[r - 3, c, plane]
        p1 = image[r - 2, c, plane]
        p0 = image[r - 1, c, plane]
        q0 = image[r, c, plane]
        q1 = image[r + 1, c, plane]
        q2 = image[r + 2, c, plane]

    # シンプルな6タップフィルターの例
    delta = <int>(round((q0 - p0) * alpha)) # alphaに基づいて差分を調整
    delta = <int>(fmax(-strength, fmin(strength, delta))) # 強度でクリップ

    cdef float beta_factor1 = 1.0 - beta * 0.5 # betaが大きいほど、境界から1つ離れた画素への影響が小さくなる
    cdef float beta_factor2 = 1.0 - beta # betaが大きいほど、境界から2つ離れた画素への影響が小さくなる

    filtered_p0 = p0 + delta
    filtered_q0 = q0 - delta
    filtered_p1 = p1 + <int>(round(delta * beta_factor1))
    filtered_q1 = q1 - <int>(round(delta * beta_factor1))
    filtered_p2 = p2 + <int>(round(delta * beta_factor2))
    filtered_q2 = q2 - <int>(round(delta * beta_factor2))

    # 結果を元の画素値の範囲にクランプ
    filtered_p2 = <int>(fmax(0.0, fmin(65535.0, filtered_p2)))
    filtered_p1 = <int>(fmax(0.0, fmin(65535.0, filtered_p1)))
    filtered_p0 = <int>(fmax(0.0, fmin(65535.0, filtered_p0)))
    filtered_q0 = <int>(fmax(0.0, fmin(65535.0, filtered_q0)))
    filtered_q1 = <int>(fmax(0.0, fmin(65535.0, filtered_q1)))
    filtered_q2 = <int>(fmax(0.0, fmin(65535.0, filtered_q2)))


    if is_horizontal:
        image[r, c - 3, plane] = filtered_p2
        image[r, c - 2, plane] = filtered_p1
        image[r, c - 1, plane] = filtered_p0
        image[r, c, plane] = filtered_q0
        image[r, c + 1, plane] = filtered_q1
        image[r, c + 2, plane] = filtered_q2
    else:
        image[r - 3, c, plane] = filtered_p2
        image[r - 2, c, plane] = filtered_p1
        image[r - 1, c, plane] = filtered_p0
        image[r, c, plane] = filtered_q0
        image[r + 1, c, plane] = filtered_q1
        image[r + 2, c, plane] = filtered_q2

# 8タップフィルター
cdef inline void apply_filter_8tap(unsigned short[:, :, :] image, int r, int c, int plane, bint is_horizontal, float strength, float alpha, float beta) nogil:
    # 8タップフィルターは境界を挟んだ8画素に適用
    # 例: 水平方向の場合、image[r, c-4] ～ image[r, c+3]
    cdef int p3, p2, p1, p0, q0, q1, q2, q3
    cdef int delta
    cdef int filtered_p3, filtered_p2, filtered_p1, filtered_p0, filtered_q0, filtered_q1, filtered_q2, filtered_q3

    if is_horizontal:
        if c < 4 or c >= image.shape[1] - 3: return # 境界の端では適用しない
        p3 = image[r, c - 4, plane]
        p2 = image[r, c - 3, plane]
        p1 = image[r, c - 2, plane]
        p0 = image[r, c - 1, plane]
        q0 = image[r, c, plane]
        q1 = image[r, c + 1, plane]
        q2 = image[r, c + 2, plane]
        q3 = image[r, c + 3, plane]
    else: # 垂直方向
        if r < 4 or r >= image.shape[0] - 3: return # 境界の端では適用しない
        p3 = image[r - 4, c, plane]
        p2 = image[r - 3, c, plane]
        p1 = image[r - 2, c, plane]
        p0 = image[r - 1, c, plane]
        q0 = image[r, c, plane]
        q1 = image[r + 1, c, plane]
        q2 = image[r + 2, c, plane]
        q3 = image[r + 3, c, plane]

    # シンプルな8タップフィルターの例
    delta = <int>(round((q0 - p0) * alpha)) # alphaに基づいて差分を調整
    delta = <int>(fmax(-strength, fmin(strength, delta))) # 強度でクリップ

    cdef float beta_factor1 = 1.0 - beta * 0.75 # betaが大きいほど、境界から1つ離れた画素への影響が小さくなる
    cdef float beta_factor2 = 1.0 - beta * 0.5  # betaが大きいほど、境界から2つ離れた画素への影響が小さくなる
    cdef float beta_factor3 = 1.0 - beta * 0.25 # betaが大きいほど、境界から3つ離れた画素への影響が小さくなる


    filtered_p0 = p0 + delta
    filtered_q0 = q0 - delta
    filtered_p1 = p1 + <int>(round(delta * beta_factor1))
    filtered_q1 = q1 - <int>(round(delta * beta_factor1))
    filtered_p2 = p2 + <int>(round(delta * beta_factor2))
    filtered_q2 = q2 - <int>(round(delta * beta_factor2))
    filtered_p3 = p3 + <int>(round(delta * beta_factor3))
    filtered_q3 = q3 - <int>(round(delta * beta_factor3))

    # 結果を元の画素値の範囲にクランプ
    filtered_p3 = <int>(fmax(0.0, fmin(65535.0, filtered_p3)))
    filtered_p2 = <int>(fmax(0.0, fmin(65535.0, filtered_p2)))
    filtered_p1 = <int>(fmax(0.0, fmin(65535.0, filtered_p1)))
    filtered_p0 = <int>(fmax(0.0, fmin(65535.0, filtered_p0)))
    filtered_q0 = <int>(fmax(0.0, fmin(65535.0, filtered_q0)))
    filtered_q1 = <int>(fmax(0.0, fmin(65535.0, filtered_q1)))
    filtered_q2 = <int>(fmax(0.0, fmin(65535.0, filtered_q2)))
    filtered_q3 = <int>(fmax(0.0, fmin(65535.0, filtered_q3)))


    if is_horizontal:
        image[r, c - 4, plane] = filtered_p3
        image[r, c - 3, plane] = filtered_p2
        image[r, c - 2, plane] = filtered_p1
        image[r, c - 1, plane] = filtered_p0
        image[r, c, plane] = filtered_q0
        image[r, c + 1, plane] = filtered_q1
        image[r, c + 2, plane] = filtered_q2
        image[r, c + 3, plane] = filtered_q3
    else:
        image[r - 4, c, plane] = filtered_p3
        image[r - 3, c, plane] = filtered_p2
        image[r - 2, c, plane] = filtered_p1
        image[r - 1, c, plane] = filtered_p0
        image[r, c, plane] = filtered_q0
        image[r + 1, c, plane] = filtered_q1
        image[r + 2, c, plane] = filtered_q2
        image[r + 3, c, plane] = filtered_q3


# メインのデブロッキング関数
cdef inline void deblock_image(unsigned short[:, :, :] image_in, unsigned short[:, :, :] image_out) nogil:
    cdef int height = image_in.shape[0]
    cdef int width = image_in.shape[1]
    cdef int planes = image_in.shape[2]

    cdef int r, c, plane
    cdef float boundary_diff, texture
    cdef int tap_count
    cdef float strength, alpha, beta

    # 入力画像をそのまま出力画像にコピー（フィルターしない画素のため）
    for plane in prange(planes, nogil=True):
        for r in prange(height):
            for c in prange(width):
                image_out[r, c, plane] = image_in[r, c, plane]

    # 水平方向の境界を処理
    # prangeを使用して平面、行、ブロック列を並列処理
    for plane in prange(planes, nogil=True):
        for r in prange(height):
            # 8ピクセルごとのブロック境界を処理
            for c in prange(BLOCK_SIZE, width, BLOCK_SIZE):
                 # 横の端4ピクセルでは縦方向のフィルターのみ処理するため、ここではスキップ
                if r < BORDER_SIZE or r >= height - BORDER_SIZE: continue

                # HVSメトリクスを計算
                boundary_diff = calculate_boundary_diff(image_in, r, c, plane, True)
                texture = calculate_local_texture(image_in, r, c, plane, True)

                # パラメータを決定
                tap_count = determine_tap_count(boundary_diff, texture, plane)
                strength = determine_filter_strength(boundary_diff, texture, plane)
                alpha = determine_alpha(boundary_diff, texture, strength, plane)
                beta = determine_beta(boundary_diff, texture, strength, plane)

                # 決定したタップ数に応じてフィルターを適用
                if tap_count == 2: apply_filter_2tap(image_out, r, c, plane, True, strength, alpha, beta)
                elif tap_count == 4: apply_filter_4tap(image_out, r, c, plane, True, strength, alpha, beta)
                elif tap_count == 6: apply_filter_6tap(image_out, r, c, plane, True, strength, alpha, beta)
                elif tap_count == 8: apply_filter_8tap(image_out, r, c, plane, True, strength, alpha, beta)
                # tap_count == 0 の場合はフィルターを適用しない (コピーされたまま)


    # 垂直方向の境界を処理
    # prangeを使用して平面、ブロック行、列を並列処理
    for plane in prange(planes, nogil=True):
        # 8ピクセルごとのブロック境界を処理
        for r in prange(BLOCK_SIZE, height, BLOCK_SIZE):
            for c in prange(width):
                # 縦の端4ピクセルでは横方向のフィルターのみ処理するため、ここではスキップ
                if c < BORDER_SIZE or c >= width - BORDER_SIZE: continue

                # HVSメトリクスを計算
                boundary_diff = calculate_boundary_diff(image_in, r, c, plane, False)
                texture = calculate_local_texture(image_in, r, c, plane, False)

                # パラメータを決定
                tap_count = determine_tap_count(boundary_diff, texture, plane)
                strength = determine_filter_strength(boundary_diff, texture, plane)
                alpha = determine_alpha(boundary_diff, texture, strength, plane)
                beta = determine_beta(boundary_diff, texture, strength, plane)

                # 決定したタップ数に応じてフィルターを適用
                if tap_count == 2: apply_filter_2tap(image_out, r, c, plane, False, strength, alpha, beta)
                elif tap_count == 4: apply_filter_4tap(image_out, r, c, plane, False, strength, alpha, beta)
                elif tap_count == 6: apply_filter_6tap(image_out, r, c, plane, False, strength, alpha, beta)
                elif tap_count == 8: apply_filter_8tap(image_out, r, c, plane, False, strength, alpha, beta)
                # tap_count == 0 の場合はフィルターを適用しない (コピーされたまま)

# この関数をPythonから呼び出すためのラッパー関数（必要に応じて）
# def apply_deblocking(image_in_np):
#     # NumPy配列をCythonメモリビューに変換
#     cdef unsigned short[:, :, :] image_in_memview = image_in_np
#     cdef unsigned short[:, :, :] image_out_memview = np.empty_like(image_in_np)
#
#     # Cython関数を呼び出し
#     deblock_image(image_in_memview, image_out_memview)
#
#     # 結果のNumPy配列を返す
#     return np.asarray(image_out_memview)

