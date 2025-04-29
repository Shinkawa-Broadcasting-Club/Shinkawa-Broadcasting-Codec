# idct.pyx
# cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np

# 以下は JPEG の ifast アルゴリズム（AAN 方式）で用いられる定数（浮動小数点）
cdef double FIX_0_298631336 = 0.298631336   # 0.298631336
cdef double FIX_0_390180644 = 0.390180644   # 0.390180644
cdef double FIX_0_541196100 = 0.541196100   # 0.541196100
cdef double FIX_0_765366865 = 0.765366865   # 0.765366865
cdef double FIX_0_899976223 = 0.899976223   # 0.899976223
cdef double FIX_1_175875602 = 1.175875602   # 1.175875602
cdef double FIX_1_501321110 = 1.501321110   # 1.501321110
cdef double FIX_1_847759065 = 1.847759065   # 1.847759065
cdef double FIX_1_961570560 = 1.961570560   # 1.961570560
cdef double FIX_2_053119869 = 2.053119869   # 2.053119869

def idct_8(np.ndarray[np.double_t, ndim=1] src):
    cdef np.ndarray[np.double_t, ndim=1] dst = np.empty(8, dtype=np.double)
    cdef double d0, d1, d2, d3, d4, d5, d6, d7
    cdef double tmp0, tmp1, tmp2, tmp3
    cdef double even0, even1, even2, even3
    cdef double odd0, odd1, odd2, odd3
    cdef double z1, z2, z3, z4, z5

    # 入力係数をローカル変数に展開
    d0 = src[0]
    d1 = src[1]
    d2 = src[2]
    d3 = src[3]
    d4 = src[4]
    d5 = src[5]
    d6 = src[6]
    d7 = src[7]

    tmp0 = d0 + d4
    tmp1 = d0 - d4
    z2 = d2
    z3 = d6
    z1 = (z2 + z3) * 0.541196100
    tmp2 = z1 + z3 * -1.847759065
    tmp3 = z1 + z2 * 0.765366865
    even0 = tmp0 + tmp3
    even3 = tmp0 - tmp3
    even1 = tmp1 + tmp2
    even2 = tmp1 - tmp2

    z2 = d7
    z3 = d5
    z1 = d3
    z4 = d1
    tmp0 = z2 + z3
    tmp1 = z1 + z4
    z5 = (tmp0 + tmp1) * 1.175875602
    tmp0 = d7 * -0.390180644 + z5
    tmp1 = d5 * -1.961570560 + z5
    tmp2 = d3 * 0.298631336 + z5
    tmp3 = d1 * 2.053119869 + z5
    odd0 = tmp0 + tmp2
    odd1 = tmp1 + tmp3
    odd2 = tmp1 - tmp3
    odd3 = tmp0 - tmp2

    dst[0] = even0 + odd0
    dst[7] = even0 - odd0
    dst[1] = even1 + odd1
    dst[6] = even1 - odd1
    dst[2] = even2 + odd2
    dst[5] = even2 - odd2
    dst[3] = even3 + odd3
    dst[4] = even3 - odd3

    # JPEG の ifast アルゴリズムでは最終的に右シフト相当のスケーリング（1/8倍）を行う
    for i in range(8): dst[i] *= 0.125
    return dst
