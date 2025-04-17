# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np
np.import_array()
from libc.string cimport memcpy
from numpy cimport PyArray_Descr, npy_intp
from numpy.core.multiarray import PyArray_NewFromDescr
from cpython.long cimport PyLong_AsVoidPtr
cdef inline int bytes_per_pixel(int bits_per_sample): return bits_per_sample // 8

cdef inline np.ndarray _wrap_plane_to_ndarray(object frame, int plane):
    cdef:
        int width   = frame.get_width(plane)
        int height  = frame.get_height(plane)
        int stride  = frame.get_stride(plane)
        int bits    = frame.format.bits_per_sample
        int bpp     = bytes_per_pixel(bits)
        int typenum
    typenum = np.NPY_UINT8 if bits <= 8 else np.NPY_UINT16
    cdef np.npy_intp dims[2]
    dims[0] = height
    dims[1] = width
    cdef np.npy_intp strides[2]
    strides[0] = stride
    strides[1] = bpp
    cdef:
        void* data_ptr = PyLong_AsVoidPtr(frame.get_read_ptr(plane))
        object descr_obj = np.PyArray_DescrFromType(typenum)
        PyArray_Descr* descr = <PyArray_Descr*>descr_obj
    if descr is NULL: raise MemoryError("NumPy dtype descriptor の取得に失敗しました。")
    cdef object arr_obj = PyArray_NewFromDescr(&np.PyArray_Type, descr, 2, dims, strides, data_ptr, np.NPY_ARRAY_ALIGNED, frame)
    if arr_obj is None: raise MemoryError("ゼロコピーによる NumPy 配列作成に失敗しました。")
    return arr_obj

cdef inline frame_to_ndarrays(object frame):
    if frame.format.num_planes < 3: raise ValueError("渡されたフレームは 3 平面 (YUV) 形式ではありません。")
    cdef:
        np.ndarray plane_y = _wrap_plane_to_ndarray(frame, 0)
        np.ndarray plane_u = _wrap_plane_to_ndarray(frame, 1)
        np.ndarray plane_v = _wrap_plane_to_ndarray(frame, 2)
    return plane_y, plane_u, plane_v

cpdef inline videonode_to_3d_ndarrays(object node, int start=0, int end=-1):
    cdef int total_frames = node.num_frames
    if end < 0 or end > total_frames: end = total_frames
    cdef:
        list y_list = []
        list u_list = []
        list v_list = []
        int i
        object frame
        np.ndarray plane_y, plane_u, plane_v
    for i in range(start, end):
         frame = node.get_frame(i)
         plane_y, plane_u, plane_v = frame_to_ndarrays(frame)
         y_list.append(plane_y)
         u_list.append(plane_u)
         v_list.append(plane_v)
         frame.free()
    cdef:
        np.ndarray y_array = np.stack(y_list, axis=0)
        np.ndarray u_array = np.stack(u_list, axis=0)
        np.ndarray v_array = np.stack(v_list, axis=0)
    return y_array, u_array, v_array