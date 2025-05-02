# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport cython

# MSVC固有のビットスキャン関数をインクルード
# intrin.h の _BitScanReverse は、unsigned long* に結果を格納し非0なら成功
cdef extern from "intrin.h":
    unsigned char _BitScanReverse(unsigned long *Index, unsigned long Mask)

@cython.inline
cdef int int_log2(unsigned int value):
    """
    MSVCの _BitScanReverse を利用して、value の整数部分の log2 を返す関数
    （value が 0 の場合は 0 を返す）
    """
    cdef unsigned long index = 0
    if _BitScanReverse(&index, <unsigned long> value):
        return <int> index
    return 0

@cython.cfunc
cdef int determinant3x3(int[:] *mat) nogil:
    """
    3x3行列のdeterminantを計算する関数
    行列は 2次元メモリビューとして渡す前提（Cythonの高速な算出）。
    """
    cdef int a = mat[0][0], b = mat[0][1], c = mat[0][2]
    cdef int d = mat[1][0], e = mat[1][1], f = mat[1][2]
    cdef int g = mat[2][0], h = mat[2][1], i = mat[2][2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_mutual_information(np.ndarray[np.int32_t, ndim=2] data):
    """
    入力: 6 x N の 32bit整数型 NumPy 配列
    手順:
      1. N個のうち無作為に256個のサンプル（列）を抽出
      2. 各変数（行）の平均を prange を用いたツリーリダクションで計算
      3. サンプルごとに (x - mean) の積の和を 256で割ることで、
         6×6 対称共分散行列を整数演算で求める
      4. 共分散行列は、変数0～2と3～5に分解し、各3×3の行列の determinant を
         計算（計算には簡易な直接式を用いる）
      5. 各変数の分散（共分散行列の対角成分）の積 prod_var と、
         近似全体の determinant （det1 * det2）との差の log2 を
         _BitScanReverse ベースの整数近似で取得し、その半分を mutual information として算出
      6. 最終的に 0 ~ 100 の範囲に収まるようにクリッピングして返す

    ※完全整数による近似計算です。演算順序・分割方法は高速化と並列化を意識した一例です。
    """
    cdef int i, j, k
    cdef int num_vars = 6
    cdef int sample_size = 256
    cdef int N = data.shape[1]
    if N < sample_size:
        raise ValueError("入力のサンプル数が不足しています")
        
    # 256個のサンプルインデックスを無作為抽出
    indices = np.random.choice(N, sample_size, replace=False)
    cdef np.ndarray[np.int32_t, ndim=2] sample = data[:, indices]
    
    # 各変数の和をツリーリダクション(prange)で計算
    cdef np.int64_t[::1] sum_arr = np.zeros(num_vars, dtype=np.int64)
    with nogil:
        for i in prange(num_vars, schedule='static'):
            cdef np.int64_t s = 0
            # nogil下での単純なforループ（ツリーリダクションの形として解釈）
            for j in range(sample_size):
                s += sample[i, j]
            sum_arr[i] = s
    cdef int means[6]
    for i in range(num_vars):
        means[i] = <int>(sum_arr[i] // sample_size)
    
    # 6x6共分散行列の算出（対称性を活かす）
    cdef int cov[6][6]
    for i in range(num_vars):
        for j in range(num_vars):
            cov[i][j] = 0
    with nogil:
        for i in prange(num_vars, schedule='static'):
            for k in range(i, num_vars):
                cdef int s = 0
                for j in range(sample_size):
                    s += (sample[i, j] - means[i]) * (sample[k, j] - means[k])
                s //= sample_size  # 整数除算
                cov[i][k] = s
                cov[k][i] = s  # 対称性
    
    # 共分散行列を、変数0～2の3x3行列と変数3～5の3x3行列に分解
    cdef int cov1[3][3]
    cdef int cov2[3][3]
    for i in range(3):
        for j in range(3):
            cov1[i][j] = cov[i][j]
            cov2[i][j] = cov[i+3][j+3]
    
    # 3x3行列のdeterminantを計算するため、一旦 NumPy の小行列にマッピング
    cdef np.ndarray[np.int32_t, ndim=2] mat1 = np.empty((3,3), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=2] mat2 = np.empty((3,3), dtype=np.int32)
    for i in range(3):
        for j in range(3):
            mat1[i, j] = cov1[i][j]
            mat2[i, j] = cov2[i][j]
    cdef int det1 = determinant3x3(mat1)  # 変数0～2の共分散部分のdeterminant
    cdef int det2 = determinant3x3(mat2)  # 変数3～5の共分散部分のdeterminant
    
    # 6変数全体の近似determinantとして、各ブロックの積を採用
    cdef int approx_det = det1 * det2
    if approx_det <= 0:
        approx_det = 1  # 非正値対策
    
    # 独立なら各変数の分散 = 共分散行列対角成分の積を用いる
    cdef int prod_var = 1
    for i in range(num_vars):
        prod_var *= (cov[i][i] if cov[i][i] > 0 else 1)
    
    # 互情報量の近似:
    # I = 0.5 * [ log2(prod_var) - log2(approx_det) ]
    cdef int log_prod_var = int_log2(prod_var)
    cdef int log_approx_det = int_log2(approx_det)
    cdef int mi = (log_prod_var - log_approx_det) // 2
    # 結果を0～100にクリッピング
    if mi < 0:
        mi = 0
    if mi > 100:
        mi = 100
    return mi
