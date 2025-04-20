# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
from cython.parallel import prange
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
from libc.stddef cimport size_t
import array
from cpython.long cimport PyLong_AsVoidPtr

cdef inline tuple vs_to_np(frames):
	cdef int nframes = frames.num_frames
	if nframes == 0: return ()
	cdef:
		object first_frame = frames[0]
		int num_planes = first_frame.format.num_planes
		int bits_per_sample = first_frame.format.bits_per_sample
		size_t element_size
	if bits_per_sample == 8: element_size = 1
	elif bits_per_sample == 16: element_size = 2
	elif bits_per_sample == 32: element_size = 4
	else: raise ValueError("Unsupported bit depth: %d" % bits_per_sample)
	cdef:
		list result_planes = [None] * num_planes
		int plane, height, width, i, j
		Py_ssize_t total_elements, total_bytes
		void **src_ptrs = NULL
		int *src_pitches = NULL
		char *dst
		size_t dst_offset
		object buf_obj, mv, ptr_obj
	for plane in range(num_planes):
		height = first_frame.get_height(plane)
		width  = first_frame.get_width(plane)
		total_elements = nframes * height * width
		total_bytes = total_elements * element_size
		if bits_per_sample == 8: buf_obj = bytearray(total_bytes)
		elif bits_per_sample == 16: buf_obj = array.array('H', [0]) * total_elements
		elif bits_per_sample == 32: buf_obj = array.array('f', [0.0]) * total_elements
		mv = memoryview(buf_obj)
		if bits_per_sample == 8: mv = mv.cast('B')
		elif bits_per_sample == 16: mv = mv.cast('H')
		elif bits_per_sample == 32: mv = mv.cast('f')
		try: mv = mv.reshape((nframes, height, width))
		except Exception as e: raise ValueError("Reshape failed: " + str(e))
		result_planes[plane] = mv
		src_ptrs = <void **> malloc(nframes * sizeof(void *))
		src_pitches = <int*> malloc(nframes * sizeof(int))
		if src_ptrs == NULL or src_pitches == NULL:
			if src_ptrs: free(src_ptrs)
			if src_pitches: free(src_pitches)
			raise MemoryError("Failed to allocate source pointer arrays")
		for i in range(nframes):
			ptr_obj = frames[i].get_read_ptr(plane)
			src_ptrs[i] = PyLong_AsVoidPtr(ptr_obj)
			src_pitches[i] = frames[i].get_stride(plane)
		dst = <char*> mv.data
		for i in prange(nframes, schedule='static', nogil=True):
			dst_offset = i * height * width * element_size
			for j in range(height): memcpy(dst + dst_offset + j * width * element_size, <char*>src_ptrs[i] + j * src_pitches[i], width * element_size)
		free(src_ptrs)
		free(src_pitches)
	return tuple(result_planes)
