# cython: language_level=3
from libc.string cimport memcpy
from cpython.bytearray cimport PyByteArray_AsString
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.parallel import prange
import cython

# --- VapourSynth 側の構造体（例） ---
cdef struct VSFormat:
    int sample_type      # 0: 整数, 1: 浮動小数点
    int bits_per_sample  # 8,16,32,64 など
    int num_planes       # 平面数（例：YUVは3）

cdef struct VSFrame:
    VSFormat* format
    void* data[4]   # 最大4平面まで
    int stride[4]   # 各平面1ラインあたりのバイト数
    int width[4]    # 各平面のピクセル幅
    int height[4]   # 各平面のピクセル高さ
# --------------------------------------------------

@cython.inline
cdef char get_format_code(VSFormat* fmt):
    """
    ピクセルフォーマットに合わせたメモリビュー用のキャストタイプコードを返す。
      - 整数の場合、8bit→b'B'[0]、16bit→b'H'[0]、32bit→b'I'[0]
      - 浮動小数点の場合、32bit→b'f'[0]、64bit→b'd'[0]
    """
    if fmt.sample_type == 0:
        if fmt.bits_per_sample == 8:
            return b'B'[0]
        elif fmt.bits_per_sample == 16:
            return b'H'[0]
        elif fmt.bits_per_sample == 32:
            return b'I'[0]
        else:
            raise ValueError("Unsupported integer bit depth")
    elif fmt.sample_type == 1:
        if fmt.bits_per_sample == 32:
            return b'f'[0]
        elif fmt.bits_per_sample == 64:
            return b'd'[0]
        else:
            raise ValueError("Unsupported float bit depth")
    else:
        raise ValueError("Unknown sample type")

cpdef tuple extract_frames_planes(tuple frames):
    """
    複数の VapourSynth フレーム（VSFrame ポインタが入ったタプル）から、
    各平面を bytearray 経由で一括コピーし、各平面を 3 次元のメモリビュー（形状 [高さ, 幅, 1]）
    として返します。
    
    処理の流れ:
      1. 各フレームごとに、各平面分のコピー先バッファ（bytearray）を確保し、
         ユーザ側で後でメモリビューとするためのリスト（buffer_list）に保持。
      2. 各平面について、ソースポインタ、コピー行サイズ、ストライド、平面サイズ等を
         C 配列に収集（全フレーム分の total_planes 個分）。
      3. [^cython.parallel.prange^] を用いた with nogil ブロック内で、
         全平面に対して行単位の memcpy コピーを並列実行。
      4. バッファからメモリビューを生成、各フレーム毎に平面タプルへまとめ返却。
    
    戻り値:
         タプル( frame0_planes, frame1_planes, ... )
         各 frameX_planes は、各平面の 3 次元メモリビューのタプルです。
    """
    cdef int nframes = len(frames)
    cdef int i, plane, plane_index = 0, num_planes, pixel_bytes, w, h, rb, total_bytes
    cdef VSFrame* cframe

    # 全フレームの平面数合計を計算
    cdef int total_planes = 0
    for i in range(nframes):
        cframe = <VSFrame*>frames[i]
        total_planes += cframe.format.num_planes

    # 各平面のパラメータ保持用 C 配列（各要素数 = total_planes）
    cdef unsigned char** src_ptrs = <unsigned char**> PyMem_Malloc(total_planes * sizeof(unsigned char*))
    cdef unsigned char** dst_ptrs = <unsigned char**> PyMem_Malloc(total_planes * sizeof(unsigned char*))
    cdef int* strides_arr = <int*> PyMem_Malloc(total_planes * sizeof(int))
    cdef Py_ssize_t* row_bytes_arr = <Py_ssize_t*> PyMem_Malloc(total_planes * sizeof(Py_ssize_t))
    cdef int* heights_arr = <int*> PyMem_Malloc(total_planes * sizeof(int))
    cdef int* widths_arr = <int*> PyMem_Malloc(total_planes * sizeof(int))
    # フレーム毎の平面数（戻り値組み立て用）
    cdef int* planes_count = <int*> PyMem_Malloc(nframes * sizeof(int))
    if (src_ptrs is NULL or dst_ptrs is NULL or strides_arr is NULL or row_bytes_arr is NULL or
        heights_arr is NULL or widths_arr is NULL or planes_count is NULL):
        raise MemoryError("Failed to allocate memory for plane parameters")
    
    # Python側のバッファは、各フレームごとに平面リストとして保持
    cdef list buffer_list = []
    # フレーム毎のメモリビューキャスト用フォーマットコードを保持
    cdef list fmt_codes = []

    # 各フレーム・各平面ごとに、コピー先の bytearray を確保しパラメータを取得
    for i in range(nframes):
        cframe = <VSFrame*>frames[i]
        num_planes = cframe.format.num_planes
        planes_count[i] = num_planes
        pixel_bytes = cframe.format.bits_per_sample // 8
        fmt_codes.append(get_format_code(cframe.format))
        buffer_list.append([])  # このフレームの平面バッファリスト
        for plane in range(num_planes):
            w = cframe.width[plane]
            h = cframe.height[plane]
            rb = w * pixel_bytes
            total_bytes = h * rb
            buffer_list[i].append(bytearray(total_bytes))
            dst_ptrs[plane_index] = <unsigned char*> PyByteArray_AsString(buffer_list[i][plane])
            src_ptrs[plane_index] = <unsigned char*> cframe.data[plane]
            strides_arr[plane_index] = cframe.stride[plane]
            row_bytes_arr[plane_index] = rb
            heights_arr[plane_index] = h
            widths_arr[plane_index] = w
            plane_index += 1

    # すべての平面のコピー処理を、prange を用いて GIL 解放下で並列実行
    cdef int p, y, ph, rb_val
    with nogil:
        # total_planes 個の各平面について独立に memcpy を実行
        for p in prange(total_planes, schedule='static'):
            ph = heights_arr[p]
            rb_val = row_bytes_arr[p]
            for y in range(ph):
                memcpy(dst_ptrs[p] + y * rb_val,
                       src_ptrs[p] + y * strides_arr[p],
                       rb_val)
    
    # コピー完了後、各 bytearray からメモリビューを生成し、各フレーム毎に平面タプルへまとめる
    cdef list result = []
    plane_index = 0
    cdef list plane_views
    cdef object mv, mv_cast
    for i in range(nframes):
        num_planes = planes_count[i]
        plane_views = []
        for j in range(num_planes):
            mv = memoryview(buffer_list[i][j])
            try:
                mv_cast = mv.cast(fmt_codes[i], shape=[heights_arr[plane_index],
                                                         widths_arr[plane_index],
                                                         1])
            except Exception as e:
                PyMem_Free(src_ptrs)
                PyMem_Free(dst_ptrs)
                PyMem_Free(strides_arr)
                PyMem_Free(row_bytes_arr)
                PyMem_Free(heights_arr)
                PyMem_Free(widths_arr)
                PyMem_Free(planes_count)
                raise ValueError("Error casting memoryview for frame %d, plane %d: %s" % (i, j, e))
            plane_views.append(mv_cast)
            plane_index += 1
        result.append(tuple(plane_views))
    
    # 確保した C メモリを解放
    PyMem_Free(src_ptrs)
    PyMem_Free(dst_ptrs)
    PyMem_Free(strides_arr)
    PyMem_Free(row_bytes_arr)
    PyMem_Free(heights_arr)
    PyMem_Free(widths_arr)
    PyMem_Free(planes_count)
    
    return tuple(result)
