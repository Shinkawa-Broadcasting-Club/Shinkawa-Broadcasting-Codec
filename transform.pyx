# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False
from cython.parallel import prange
cimport numpy as cnp
import numpy as np

# Helper function to perform a signed right shift by 8 bits
cdef inline int sft8_sgn(int n) nogil:
    if n > 0x7FFFFFFF: # Check for negative numbers represented as large unsigned
        return -(-n >> 8)
    else:
        return n >> 8

# 1D Forward Discrete Cosine Transform (DCT) on an 8-element block
# i_ptr: Pointer to the start of the 8-element block
# step: Stride between elements (e.g., 1 for row, 8 for column, 64 for depth in an 8x8x8 block)
cdef inline void dct_1d_fwd(int* i_ptr, int step) nogil:
    cdef int* i = <int*>i_ptr
    cdef int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tmp16, z1, z2, z3, z4, z5, z11, z13

    # Stage 1
    tmp0 = i[0 * step] + i[7 * step]
    tmp7 = i[0 * step] - i[7 * step]
    tmp1 = i[1 * step] + i[6 * step]
    tmp6 = i[1 * step] - i[6 * step]
    tmp2 = i[2 * step] + i[5 * step]
    tmp5 = i[2 * step] - i[5 * step]
    tmp3 = i[3 * step] + i[4 * step]
    tmp4 = i[3 * step] - i[4 * step]

    # Stage 2
    tmp10 = tmp0 + tmp3
    tmp11 = tmp1 + tmp2
    tmp12 = tmp1 - tmp2
    tmp13 = tmp0 - tmp3
    tmp14 = tmp4 + tmp5
    tmp15 = tmp5 + tmp6
    tmp16 = tmp6 + tmp7

    # Stage 3 (Rotations/Multiplications)
    # These constants (181, 98, 139, 334) are approximations scaled for fixed-point arithmetic
    z1 = sft8_sgn((tmp12 + tmp13) * 181)
    z3 = sft8_sgn(tmp15 * 181)
    z5 = (tmp14 - tmp16) * 98

    z2 = sft8_sgn(tmp14 * 139 + z5)
    z4 = sft8_sgn(tmp16 * 334 + z5)
    z11 = tmp7 + z3
    z13 = tmp7 - z3

    # Stage 4 (Final additions/subtractions)
    i[0 * step] = tmp10 + tmp11
    i[1 * step] = z11 + z4
    i[2 * step] = tmp13 + z1
    i[3 * step] = z13 - z2
    i[4 * step] = tmp10 - tmp11
    i[5 * step] = z13 + z2
    i[6 * step] = tmp13 - z1
    i[7 * step] = z11 - z4

# 1D Backward Discrete Cosine Transform (DCT) on an 8-element block
# i_ptr: Pointer to the start of the 8-element block
# step: Stride between elements
cdef inline void dct_1d_bwd(int* i_ptr, int step) nogil:
    cdef int* i = <int*>i_ptr
    cdef int x0, x1, x2, x3, x4, x5, x6, x7, a0, a1, a2, a3, a4, a5, a6, a7

    # Stage 1 (Inverse of Stage 4 in forward)
    a0 = i[0 * step] + i[4 * step]
    a1 = i[0 * step] - i[4 * step]
    a2 = i[2 * step] + i[6 * step]
    # Approximation constant 181
    a3 = sft8_sgn((i[2 * step] - i[6 * step]) * 181)
    a4 = i[1 * step] + i[7 * step]
    a5 = i[5 * step] + i[3 * step]
    a6 = i[5 * step] - i[3 * step]
    a7 = i[1 * step] - i[7 * step]

    # Stage 2 (Inverse of Stage 3 in forward)
    x0 = a0 + a2
    x2 = a0 - a2
    x4 = a1 + a3
    x6 = a1 - a3
    # Approximation constants 362, 98, 139, 334
    x1 = sft8_sgn((a4 + a5) * 362)
    x3 = sft8_sgn((a7 + a6) * 98)
    x5 = sft8_sgn((a4 - a5) * 139)
    x7 = sft8_sgn((a7 - a6) * 334)

    # Stage 3 (Inverse of Stage 2 in forward)
    i[0 * step] = x0 + x1
    i[1 * step] = x4 + x3
    i[2 * step] = x2 + x5
    i[3 * step] = x6 + x7
    i[4 * step] = x6 - x7
    i[5 * step] = x2 - x5
    i[6 * step] = x4 - x3
    i[7 * step] = x0 - x1

# Helper function for a time-domain transform (likely related to motion compensation or temporal filtering)
# This looks like a simple butterfly structure.
cdef inline void dct_time(int* i_ptr, int step) nogil:
    cdef int* i = <int*>i_ptr
    cdef int j0, j1, j2, j3, j4, j5, j6, j7, k0, k1, k2, k3, k4, k5, k6, k7

    # Stage 1
    j0 = i[0 * step] + i[4 * step]
    j1 = i[1 * step] + i[5 * step]
    j2 = i[2 * step] + i[6 * step]
    j3 = i[3 * step] + i[7 * step]
    j4 = i[0 * step] - i[4 * step]
    j5 = i[1 * step] - i[5 * step]
    j6 = i[2 * step] - i[6 * step]
    j7 = i[3 * step] - i[7 * step]

    # Stage 2
    k0 = j0 + j2
    k1 = j1 + j3
    k2 = j4 + j6
    k3 = j5 + j7
    k4 = j0 - j2
    k5 = j1 - j3
    k6 = j4 - j6
    k7 = j5 - j7

    # Stage 3
    i[0 * step] = k0 + k1
    i[1 * step] = k0 - k1
    i[2 * step] = k2 + k3
    i[3 * step] = k2 - k3
    i[4 * step] = k4 + k5
    i[5 * step] = k4 - k5
    i[6 * step] = k6 + k7
    i[7 * step] = k6 - k7

# 3D Forward transform on an 8x8x8 block (likely DCT + temporal transform)
# inp: Pointer to the start of the 8x8x8 block in memory
cdef inline void vrdif_3d_fwd(int* inp) nogil:
    cdef int l, m, n

    # Apply 1D DCT along the first dimension (e.g., rows)
    for l in prange(8):
        for m in prange(8):
            n = (l << 6) + (m << 3) # Calculate starting index for each 1D block (8 elements)
            dct_1d_fwd(&inp[n], 1) # Apply DCT with step 1

    # Apply 1D DCT along the second dimension (e.g., columns)
    for l in prange(8):
        for m in prange(8):
            n = (l << 6) + m # Calculate starting index for each 1D block (8 elements)
            dct_1d_fwd(&inp[n], 8) # Apply DCT with step 8

    # Apply time-domain transform along the third dimension (e.g., depth/time)
    for l in prange(8):
        for m in prange(8):
            n = (l << 3) + m # Calculate starting index for each 1D block (8 elements)
            dct_time(&inp[n], 64) # Apply transform with step 64

# 3D Backward transform on an 8x8x8 block
# inp: Pointer to the start of the 8x8x8 block in memory
cdef inline void vrdif_3d_bwd(int* inp) nogil:
    cdef int l, m, n

    # Apply inverse time-domain transform
    for l in prange(8):
        for m in prange(8):
            n = (l << 3) + m # Calculate starting index
            dct_time(&inp[n], 64) # Apply inverse transform

    # Apply 1D Inverse DCT along the second dimension (e.g., columns)
    for l in prange(8):
        for m in prange(8):
            n = (l << 6) + m # Calculate starting index
            dct_1d_bwd(&inp[n], 8) # Apply inverse DCT with step 8

    # Apply 1D Inverse DCT along the first dimension (e.g., rows)
    for l in prange(8):
        for m in prange(8):
            n = (l << 6) + (m << 3) # Calculate starting index
            dct_1d_bwd(&inp[n], 1) # Apply inverse DCT with step 1

# 3D Forward DCT for a larger array, processed in 8x8x8 blocks
# arr: Input NumPy array (memory view)
# Returns a new NumPy array with the transformed data
cpdef inline int[:, :, :] dct_3d_fwd(unsigned short[:, :, :] arr):
    cdef int x = <int> arr.shape[0]
    cdef int y = <int> arr.shape[1]
    cdef int z = <int> arr.shape[2]
    # Ensure array dimensions are multiples of 8
    if x & 7 != 0 or y & 7 != 0 or z & 7 != 0: raise ValueError("Array dimensions must be multiples of 8")

    # Create output array with the same shape and dtype
    cdef cnp.ndarray[cnp.int32_t, ndim=3] o = np.empty(arr.shape, np.int32)
    cdef int[:, :, :] out = o # Create a memory view for the output array

    cdef int[512] tmp # Local C array to hold an 8x8x8 block
    cdef int l, m, n, a, b, c

    # Iterate over 8x8x8 blocks
    # prange parallelizes the outermost loop (over blocks in the first dimension)
    for l in prange(x >> 3, nogil=True, schedule='static'):
        for m in prange(y >> 3):
            for n in prange(z >> 3):
                # Copy data from the input array block to the local C array
                for a in prange(8):
                    for b in prange(8):
                        for c in prange(8):
                            tmp[(a << 6) + (b << 3) + c] = arr[(l << 3) + a, (m << 3) + b, (n << 3) + c]

                # Apply the 3D forward transform on the local block
                vrdif_3d_fwd(&tmp[0]) # Pass pointer to the start of the C array

                # Copy data from the transformed local C array back to the output array block
                for a in prange(8):
                    for b in prange(8):
                        for c in prange(8):
                            out[(l << 3) + a, (m << 3) + b, (n << 3) + c] = tmp[(a << 6) + (b << 3) + c]

    return out

# 3D Backward DCT for a larger array, processed in 8x8x8 blocks
# arr: Input NumPy array (memory view)
# Returns a new NumPy array with the inverse transformed data
cpdef inline int[:, :, :] dct_3d_bwd(int[:, :, :] arr):
    cdef int x = <int> arr.shape[0]
    cdef int y = <int> arr.shape[1]
    cdef int z = <int> arr.shape[2]
    # Ensure array dimensions are multiples of 8
    if x & 7 != 0 or y & 7 != 0 or z & 7 != 0: raise ValueError("Array dimensions must be multiples of 8")

    # Create output array
    cdef cnp.ndarray[cnp.int32_t, ndim=3] o = np.empty(arr.shape, np.int32)
    cdef int[:, :, :] out = o # Memory view for output

    cdef int[512] tmp # Local C array for 8x8x8 block
    cdef int l, m, n, a, b, c

    # Iterate over 8x8x8 blocks
    # prange parallelizes the outermost loop
    for l in prange(x >> 3, nogil=True, schedule='static'):
        for m in prange(y >> 3):
            for n in prange(z >> 3):
                # Copy data from input array block to local C array
                for a in prange(8):
                    for b in prange(8):
                        for c in prange(8):
                            tmp[(a << 6) + (b << 3) + c] = arr[(l << 3) + a, (m << 3) + b, (n << 3) + c]

                # Apply the 3D backward transform on the local block
                vrdif_3d_bwd(&tmp[0]) # Pass pointer to the start of the C array

                # Copy data from transformed local C array back to output array block
                for a in prange(8):
                    for b in prange(8):
                        for c in prange(8):
                            out[(l << 3) + a, (m << 3) + b, (n << 3) + c] = tmp[(a << 6) + (b << 3) + c]

    return out
