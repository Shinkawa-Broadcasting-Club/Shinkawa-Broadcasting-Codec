# distutils: language = c++
# distutils: extra_compile_args = /O2 /openmp
# distutils: extra_link_args = /openmp

cimport cython
cimport numpy as cnp
cimport vapoursynth.vsapi as vsapi
from libc.string cimport memcpy
from cython.parallel cimport prange

cdef extern from "VapourSynth.h":
	ctypedef struct VSNode: pass
	ctypedef struct VSFrame: pass
	ctypedef struct VSCore: pass
	ctypedef struct VSApi: pass
	VSCore* getCore(VSApi*)
	const VSFrame* getFrame(int n, VSNode* node, VSCore* core) except +
	void freeFrame(const VSFrame* frame)
	const unsigned char* getReadPtr(const VSFrame* frame, int plane)
	int getStride(const VSFrame* frame, int plane)
	int getFrameWidth(const VSFrame* frame, int plane)
	int getFrameHeight(const VSFrame* frame, int plane)
	const VSFormat* getFrameFormat(const VSFrame* frame)
	const VSApi* getVSApi(int version)
	ctypedef struct VSFormat:
		int bytesPerSample
		int bitsPerSample
		int colorFamily
		int subsamplingW
		int subsamplingH
		bint isConstantFormat

	# Vapoursynthの定数を extern from ブロック内に宣言
	int cfYUV;
	int stUSHORT;

cdef extern from "numpy/arrayobject.h":
	int PyArray_API_VERSION
	int NPY_ARRAY_C_CONTIGUOUS
	int NPY_ARRAY_WRITEABLE
	int NPY_ARRAY_EMPTY
	int NPY_USHORT
	object PyArray_EMPTY(int nd, cnp.npy_intp* dims, int dtype, int flags)

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.inline(True)
cdef tuple[unsigned short[:, :, :], unsigned short[:, :, :], unsigned short[:, :, :]] get_yuv444p16_planes(vsapi.VSNode node, int num_frames):
	cdef vsapi.VSCore* core = vsapi.getCore(vsapi.getVSApi(0))
	cdef const vsapi.VSFrame* frame = NULL
	cdef const vsapi.VSFormat* format = NULL
	cdef int width, height
	cdef int frame_idx, plane_idx, row
	cdef const unsigned char* src_ptr
	cdef int src_stride
	cdef unsigned short* dest_ptr_row
	cdef int dest_stride
	cdef cnp.npy_intp dims[3]
	cdef list acquired_frames = []
	for frame_idx in range(num_frames):
		frame = core.getFrame(frame_idx, node)
		if frame is NULL:
			with gil:
				for f_obj in acquired_frames:
					vsapi.freeFrame(<const vsapi.VSFrame*> f_obj)
			raise RuntimeError(f"フレーム {frame_idx} の取得に失敗しました")
		with gil:
			acquired_frames.append(<object>frame)

	if num_frames > 0:
		with gil:
			frame = <const vsapi.VSFrame*> acquired_frames[0]

		format = vsapi.getFrameFormat(frame)
		if format.colorFamily != vsapi.cfYUV or format.bytesPerSample != 2 or format.bitsPerSample != 16 or format.subsamplingW != 0 or format.subsamplingH != 0:
			with gil:
				for f_obj in acquired_frames:
					vsapi.freeFrame(<const vsapi.VSFrame*> f_obj)
			raise ValueError("入力クリップはYUV444P16フォーマットである必要があります")

		width = vsapi.getFrameWidth(frame, 0)
		height = vsapi.getFrameHeight(frame, 0)
	else:
		with gil:
			return (<unsigned short[:, :, :]>cnp.PyArray_EMPTY(3, <cnp.npy_intp*>&([0, 0, 0])[0], cnp.NPY_USHORT, NPY_ARRAY_EMPTY | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE),
					<unsigned short[:, :, :]>cnp.PyArray_EMPTY(3, <cnp.npy_intp*>&([0, 0, 0])[0], cnp.NPY_USHORT, NPY_ARRAY_EMPTY | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE),
					<unsigned short[:, :, :]>cnp.PyArray_EMPTY(3, <cnp.npy_intp*>&([0, 0, 0])[0], cnp.NPY_USHORT, NPY_ARRAY_EMPTY | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE))

	dims[0] = num_frames
	dims[1] = height
	dims[2] = width

	cdef object y_plane_obj
	cdef object u_plane_obj
	cdef object v_plane_obj

	with gil:
		y_plane_obj = cnp.PyArray_EMPTY(3, dims, cnp.NPY_USHORT, NPY_ARRAY_EMPTY | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE)
		u_plane_obj = cnp.PyArray_EMPTY(3, dims, cnp.NPY_USHORT, NPY_ARRAY_EMPTY | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE)
		v_plane_obj = cnp.PyArray_EMPTY(3, dims, cnp.NPY_USHORT, NPY_ARRAY_EMPTY | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE)

	cdef unsigned short[:, :, :] y_plane_mv = y_plane_obj
	cdef unsigned short[:, :, :] u_plane_mv = u_plane_obj
	cdef unsigned short[:, :, :] v_plane_mv = v_plane_obj

	dest_stride = width * sizeof(unsigned short)

	cdef const unsigned char* y_src_ptr
	cdef const unsigned char* u_src_ptr
	cdef const unsigned char* v_src_ptr
	cdef int y_src_stride
	cdef int u_src_stride
	cdef int v_src_stride

	for frame_idx in prange(num_frames, nogil=True):
		cdef const vsapi.VSFrame* current_frame = NULL
		with gil:
			 current_frame = <const vsapi.VSFrame*> acquired_frames[frame_idx]

		y_src_ptr = vsapi.getReadPtr(current_frame, 0)
		y_src_stride = vsapi.getStride(current_frame, 0)
		u_src_ptr = vsapi.getReadPtr(current_frame, 1)
		u_src_stride = vsapi.getStride(current_frame, 1)
		v_src_ptr = vsapi.getReadPtr(current_frame, 2)
		v_src_stride = vsapi.getStride(current_frame, 2)

		for row in range(height):
			dest_ptr_row = &y_plane_mv[frame_idx, row, 0]
			memcpy(dest_ptr_row, y_src_ptr + row * y_src_stride, <size_t>width * sizeof(unsigned short))

			dest_ptr_row = &u_plane_mv[frame_idx, row, 0]
			memcpy(dest_ptr_row, u_src_ptr + row * u_src_stride, <size_t>width * sizeof(unsigned short))

			dest_ptr_row = &v_plane_mv[frame_idx, row, 0]
			memcpy(dest_ptr_row, v_src_ptr + row * v_src_stride, <size_t>width * sizeof(unsigned short))

	with gil:
		for f_obj in acquired_frames:
			 vsapi.freeFrame(<const vsapi.VSFrame*> f_obj)

	with gil:
		return y_plane_mv, u_plane_mv, v_plane_mv
