# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange, parallel

cdef inline calc_ssim_rd(float[:, :] arr):
    