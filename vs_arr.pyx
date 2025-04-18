# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
from libc.string cimport memcpy
from cpython.bytearray cimport PyByteArray_AsString
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.parallel import prange
import cython
cdef struct VSFormat:
	int sample_type
	int bits_per_sample
	int num_planes

cdef struct VSFrame:
	VSFormat* format
	void* data[4]
	int stride[4]
	int width[4]
	int height[4]

cdef inline char get_format_code(VSFormat* fmt):
	if fmt.sample_type == 0:
		if fmt.bits_per_sample == 8: return b'B'[0]
		elif fmt.bits_per_sample == 16: return b'H'[0]
		elif fmt.bits_per_sample == 32: return b'I'[0]
		else: raise ValueError("Unsupported integer bit depth")
	elif fmt.sample_type == 1:
		if fmt.bits_per_sample == 32: return b'f'[0]
		elif fmt.bits_per_sample == 64: return b'd'[0]
		else: raise ValueError("Unsupported float bit depth")
	else: raise ValueError("Unknown sample type")

cpdef inline tuple extract_frames_planes(tuple frames):
	cdef:
		int nframes = len(frames)
		int i, plane, plane_index = 0, num_planes, pixel_bytes, w, h, rb, total_bytes
		VSFrame* cframe
		int total_planes = 0
	for i in range(nframes):
		cframe = <VSFrame*>frames[i]
		total_planes += cframe.format.num_planes
	cdef:
		unsigned char** src_ptrs = <unsigned char**> PyMem_Malloc(total_planes * sizeof(unsigned char*))
		unsigned char** dst_ptrs = <unsigned char**> PyMem_Malloc(total_planes * sizeof(unsigned char*))
		int* strides_arr = <int*> PyMem_Malloc(total_planes * sizeof(int))
		Py_ssize_t* row_bytes_arr = <Py_ssize_t*> PyMem_Malloc(total_planes * sizeof(Py_ssize_t))
		int* heights_arr = <int*> PyMem_Malloc(total_planes * sizeof(int))
		int* widths_arr = <int*> PyMem_Malloc(total_planes * sizeof(int))
		int* planes_count = <int*> PyMem_Malloc(nframes * sizeof(int))
	if (src_ptrs is NULL or dst_ptrs is NULL or strides_arr is NULL or row_bytes_arr is NULL or heights_arr is NULL or widths_arr is NULL or planes_count is NULL): raise MemoryError("Failed to allocate memory for plane parameters")
	cdef:
		list buffer_list = []
		list fmt_codes = []
	for i in range(nframes):
		cframe = <VSFrame*>frames[i]
		num_planes = cframe.format.num_planes
		planes_count[i] = num_planes
		pixel_bytes = cframe.format.bits_per_sample // 8
		fmt_codes.append(get_format_code(cframe.format))
		buffer_list.append([])
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
	cdef int p, y, ph, rb_val
	with nogil:
		for p in prange(total_planes, schedule='static'):
			ph = heights_arr[p]
			rb_val = row_bytes_arr[p]
			for y in range(ph): memcpy(dst_ptrs[p] + y * rb_val, src_ptrs[p] + y * strides_arr[p], rb_val)
	cdef list result = []
	plane_index = 0
	cdef:
		list plane_views
		object mv, mv_cast
	for i in range(nframes):
		num_planes = planes_count[i]
		plane_views = []
		for j in range(num_planes):
			mv = memoryview(buffer_list[i][j])
			try: mv_cast = mv.cast(fmt_codes[i], shape=[heights_arr[plane_index], widths_arr[plane_index], 1])
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
	PyMem_Free(src_ptrs)
	PyMem_Free(dst_ptrs)
	PyMem_Free(strides_arr)
	PyMem_Free(row_bytes_arr)
	PyMem_Free(heights_arr)
	PyMem_Free(widths_arr)
	PyMem_Free(planes_count)
	return tuple(result)
