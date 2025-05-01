# cython: boundscheck=False, wraparound=False, nonecheck=False
from libc.stdlib cimport rand, srand, malloc, free
from libc.string cimport memcpy
from libc.time cimport time
from cython.parallel import parallel, prange, reduction
cdef extern from "intrin.h" nogil:
    unsigned char _BitScanReverse(unsigned long *Index, unsigned long Mask)

cdef inline rand256(int[:, :] arr) nogil:
    cdef:
        int N = arr.shape[1]
        int[3][256] out
        int* indices = <int*> malloc(N * sizeof(int))
        int i, j, k, tmp
    if N < 256: raise ValueError("入力配列の第2軸（列数）は256以上である必要があります。")
    if indices == NULL: raise MemoryError("インデックス配列のメモリ確保に失敗しました。")
    with nogil, parallel():
        for i in prange(N): indices[i] = i
    srand(<unsigned int> time(NULL))
    for i in range(256):
        j = i + rand() % (N - i)
        tmp = indices[i]
        indices[i] = indices[j]
        indices[j] = tmp
        for k in range(3): out[k][i] = arr[k, indices[i]]
    free(indices)
    return out

cdef inline int ilog2(unsigned int v) nogil:
    """
    v の最上位に立っているビットのインデックスを返します。
    例: 32 (0b00100000) の場合、返り値は 5 (0-indexed) になります。
    入力が 0 の場合は ValueError を発生させます。
    """
    cdef unsigned long index
    if _BitScanReverse(&index, v) == 0: raise ValueError("v must be non-zero")
    return index

cdef inline corr_coef(int[:, :] arr) nogil:
    cdef:
        int[3][256] sample = rand256(arr)
        int x = sample[0]
        int y = sample[1]
        int z = sample[2]
        unsigned short i, j
    with nogil, parallel():
        cdef int n = 256
        cdef int i
        cdef int sx = 0, sy = 0, sz = 0

        # 各配列の合計値を並列化して計算（reductionを利用）
        for i in prange(n):
            sx += x[i]
            sy += y[i]
            sz += z[i]

    # 256で割る＝右に8ビットシフト
    cdef int mean_x = sx >> 8
    cdef int mean_y = sy >> 8
    cdef int mean_z = sz >> 8

    cdef int Sxx = 0, Syy = 0, Szz = 0
    cdef int Sxy = 0, Sxz = 0, Syz = 0
    cdef int dx, dy, dz

    # 分散（Sxx,Syy,Szz）および共分散（Sxy,Sxz,Syz）の計算
    for i in prange(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        dz = z[i] - mean_z
        Sxx += dx ** 2
        Syy += dy ** 2
        Szz += dz ** 2
        Sxy += dx * dy
        Sxz += dx * dz
        Syz += dy * dz

    # 平均で割るのは右シフト8で
    cdef int cov_xx = Sxx >> 8
    cdef int cov_yy = Syy >> 8
    cdef int cov_zz = Szz >> 8
    cdef int cov_xy = Sxy >> 8
    cdef int cov_xz = Sxz >> 8
    cdef int cov_yz = Syz >> 8

    # 共分散行列の行列式を計算
    cdef int det = cov_xx * cov_yy * cov_zz - cov_xx * cov_yz ** 2 - cov_xy * cov_xy * cov_zz - cov_xz ** 2 * cov_yy + 2 * cov_xy * cov_yz * cov_xz

    # 各変量の分散の積
    cdef int prod = cov_xx * cov_yy * cov_zz

    # もし det または prod が正でなければ、計算不能と見なす
    if det <= 0 or prod <= 0: return 0

    # 対数を整数近似 (log₂の整数部分) を用いて相互情報量を計算
    cdef int log_prod = ilog2(prod)
    cdef int log_det = ilog2(det)
    return (log_prod - log_det) >> 1 # 0.5倍するため右シフト1（1/2倍）
