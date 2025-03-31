import numpy as np
import cython
cimport cython
cimport numpy as cnp
cdef extern from *:
	"""
	#include <immintrin.h>
	"""
	ctypedef struct __m512i: pass
	ctypedef struct __m512h: pass
	ctypedef struct __mmask8: pass
	__m512i _mm512_set_epi16(short e31, short e30, short e29, short e28, short e27, short e26, short e25, short e24, short e23, short e22, short e21, short e20, short e19, short e18, short e17, short e16, short e15, short e14, short e13, short e12, short e11, short e10, short e9, short e8, short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
	__m512i _mm512_mask_mov_epi64(__m512i src, __mmask8 k, __m512i a)
	__m512h _mm512_cvtepu16_ph(__m512i a)
	__m512i _mm512_castph_si512(__m512h a)
	__m512h _mm512_castsi512_ph(__m512i a)
	__m512h _mm512_add_ph(__m512h a, __m512h b)
	__m512h _mm512_sub_ph(__m512h a, __m512h b)

@cython.boundscheck(False)
@cython.wraparound(False)
def dct_time_fwd(cnp.ndarray[cnp.uint16_t, ndim = 1] a):
	cdef __mmask8 m0 = 15
	cdef __mmask8 m1 = 240
	cdef __mmask8 m2 = 51
	cdef __mmask8 m3 = 204
	# loading
	cdef __m512i v0 = _mm512_set_epi16(a[31], a[30], a[29], a[28], a[27], a[26], a[25], a[24], a[23], a[22], a[21], a[20], a[19], a[18], a[17], a[16], a[15], a[14], a[13], a[12], a[11], a[10], a[9], a[8], a[7], a[6], a[5], a[4], a[3], a[2], a[1], a[0])
	cdef __m512i v1 = _mm512_set_epi16(a[63], a[62], a[61], a[60], a[59], a[58], a[57], a[56], a[55], a[54], a[53], a[52], a[51], a[50], a[49], a[48], a[47], a[46], a[45], a[44], a[43], a[42], a[41], a[40], a[39], a[38], a[37], a[36], a[35], a[34], a[33], a[32])
	cdef __m512h v2 = _mm512_cvtepu16_ph(v0)
	cdef __m512h v3 = _mm512_cvtepu16_ph(v1)
	# stage 1
	cdef __m512h v4 = _mm512_add_ph(v2, v3)
	cdef __m512h v5 = _mm512_sub_ph(v2, v3)
	# move
	cdef __m512i v6 = _mm512_castph_si512(v4)
	cdef __m512i v7 = _mm512_castph_si512(v5)
	cdef __m512i v8 = _mm512_mask_mov_epi64(v6, m1, v7)
	cdef __m512i v9 = _mm512_mask_mov_epi64(v7, m0, v6)
	cdef __m512h v10 = _mm512_castsi512_ph(v8)
	cdef __m512h v11 = _mm512_castsi512_ph(v9)
	# stage 2
	cdef __m512h v12 = _mm512_add_ph(v10, v11)
	cdef __m512h v13 = _mm512_sub_ph(v10, v11)
	# move
	cdef __m512i v14 = _mm512_castph_si512(v12)
	cdef __m512i v15 = _mm512_castph_si512(v13)
	cdef __m512i v16 = _mm512_mask_mov_epi64(v14, m3, v15)
	cdef __m512i v17 = _mm512_mask_mov_epi64(v15, m2, v14)
	cdef __m512h v18 = _mm512_castsi512_ph(v16)
	cdef __m512h v19 = _mm512_castsi512_ph(v17)
	cdef cnp.ndarray[cnp.uint16_t, ndim = 1] result = np.empty(64, np.uint16)
	return result