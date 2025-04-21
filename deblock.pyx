# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: profile=False

from cython.parallel import prange
from libc.math cimport fabs, fmax, fmin

cdef int BLOCK_SIZE = 8
cdef int BORDER_SIZE = 4

cdef float Y_BOUNDARY_DIFF_THRESHOLD_FLAT = 8.0
cdef float Y_BOUNDARY_DIFF_THRESHOLD_EDGE = 32.0
cdef float Y_TEXTURE_THRESHOLD_FLAT = 16.0
cdef float Y_TEXTURE_THRESHOLD_TEXTURED = 64.0
cdef float Y_MAX_FILTER_STRENGTH = 64.0
cdef float Y_ALPHA_FLAT = 0.1
cdef float Y_ALPHA_TEXTURED = 0.5
cdef float Y_BETA_FLAT = 0.5
cdef float Y_BETA_TEXTURED = 0.1

cdef float UV_BOUNDARY_DIFF_THRESHOLD_FLAT = 16.0
cdef float UV_BOUNDARY_DIFF_THRESHOLD_EDGE = 64.0
cdef float UV_TEXTURE_THRESHOLD_FLAT = 32.0
cdef float UV_TEXTURE_THRESHOLD_TEXTURED = 128.0
cdef float UV_MAX_FILTER_STRENGTH = 32.0
cdef float UV_ALPHA_FLAT = 0.2
cdef float UV_ALPHA_TEXTURED = 0.6
cdef float UV_BETA_FLAT = 0.6
cdef float UV_BETA_TEXTURED = 0.2

cdef inline int determine_tap_count(float boundary_diff, float texture, int plane_idx) nogil:
    cdef float boundary_flat_thresh, boundary_edge_thresh
    cdef float texture_flat_thresh, texture_textured_thresh
    if plane_idx == 0:
        boundary_flat_thresh = Y_BOUNDARY_DIFF_THRESHOLD_FLAT
        boundary_edge_thresh = Y_BOUNDARY_DIFF_THRESHOLD_EDGE
        texture_flat_thresh = Y_TEXTURE_THRESHOLD_FLAT
        texture_textured_thresh = Y_TEXTURE_THRESHOLD_TEXTURED
    else:
        boundary_flat_thresh = UV_BOUNDARY_DIFF_THRESHOLD_FLAT
        boundary_edge_thresh = UV_BOUNDARY_DIFF_THRESHOLD_EDGE
        texture_flat_thresh = UV_TEXTURE_THRESHOLD_FLAT
        texture_textured_thresh = UV_TEXTURE_THRESHOLD_TEXTURED
    if boundary_diff > boundary_edge_thresh: return 0
    elif boundary_diff > boundary_flat_thresh:
        cdef float ratio = (boundary_diff - boundary_flat_thresh) / (boundary_edge_thresh - boundary_flat_thresh)
        cdef float texture_ratio = (texture - texture_flat_thresh) / (texture_textured_thresh - texture_flat_thresh)
        texture_ratio = fmax(0.0, fmin(1.0, texture_ratio))
        cdef float tap_ratio = fmin(ratio + texture_ratio, 1.0)
        if tap_ratio < 0.2: return 8
        elif tap_ratio < 0.4: return 6
        elif tap_ratio < 0.6: return 4
        elif tap_ratio < 0.8: return 2
        else: return 0
    else:
        if texture < texture_flat_thresh: return 8
        elif texture < texture_textured_thresh:
            cdef float ratio = (texture - texture_flat_thresh) / (texture_textured_thresh - texture_flat_thresh)
            if ratio < 0.25: return 8
            elif ratio < 0.5: return 6
            elif ratio < 0.75: return 4
            else: return 2
        else: return 0

cdef inline float determine_filter_strength(float boundary_diff, float texture, int plane_idx) nogil:
    cdef float boundary_flat_thresh, boundary_edge_thresh
    cdef float texture_flat_thresh, texture_textured_thresh
    cdef float max_strength
    if plane_idx == 0:
        boundary_flat_thresh = Y_BOUNDARY_DIFF_THRESHOLD_FLAT
        boundary_edge_thresh = Y_BOUNDARY_DIFF_THRESHOLD_EDGE
        texture_flat_thresh = Y_TEXTURE_THRESHOLD_FLAT
        texture_textured_thresh = Y_TEXTURE_THRESHOLD_TEXTURED
        max_strength = Y_MAX_FILTER_STRENGTH
    else:
        boundary_flat_thresh = UV_BOUNDARY_DIFF_THRESHOLD_FLAT
        boundary_edge_thresh = UV_BOUNDARY_DIFF_THRESHOLD_EDGE
        texture_flat_thresh = UV_TEXTURE_THRESHOLD_FLAT
        texture_textured_thresh = UV_TEXTURE_THRESHOLD_TEXTURED
        max_strength = UV_MAX_FILTER_STRENGTH
    cdef float boundary_ratio = 1.0 - fmax(0.0, fmin(1.0, (boundary_diff - boundary_flat_thresh) / (boundary_edge_thresh - boundary_flat_thresh)))
    cdef float texture_ratio = 1.0 - fmax(0.0, fmin(1.0, (texture - texture_flat_thresh) / (texture_textured_thresh - texture_textured_thresh)))
    cdef float strength_ratio = (boundary_ratio + texture_ratio) / 2.0
    return max_strength * strength_ratio

cdef inline float determine_alpha(float boundary_diff, float texture, float strength, int plane_idx) nogil:
    cdef float boundary_flat_thresh, boundary_edge_thresh
    cdef float texture_flat_thresh, texture_textured_thresh
    cdef float alpha_flat, alpha_textured
    if plane_idx == 0:
        boundary_flat_thresh = Y_BOUNDARY_DIFF_THRESHOLD_FLAT
        boundary_edge_thresh = Y_BOUNDARY_DIFF_THRESHOLD_EDGE
        texture_flat_thresh = Y_TEXTURE_THRESHOLD_FLAT
        texture_textured_thresh = Y_TEXTURE_THRESHOLD_TEXTURED
        alpha_flat = Y_ALPHA_FLAT
        alpha_textured = Y_ALPHA_TEXTURED
    else:
        boundary_flat_thresh = UV_BOUNDARY_DIFF_THRESHOLD_FLAT
        boundary_edge_thresh = UV_BOUNDARY_DIFF_THRESHOLD_EDGE
        texture_flat_thresh = UV_TEXTURE_THRESHOLD_FLAT
        texture_textured_thresh = UV_TEXTURE_THRESHOLD_TEXTURED
        alpha_flat = UV_ALPHA_FLAT
        alpha_textured = UV_ALPHA_TEXTURED
    cdef float texture_ratio = fmax(0.0, fmin(1.0, (texture - texture_flat_thresh) / (texture_textured_thresh - texture_flat_thresh)))
    cdef float alpha = alpha_flat + (alpha_textured - alpha_flat) * texture_ratio
    cdef float max_strength = Y_MAX_FILTER_STRENGTH if plane_idx == 0 else UV_MAX_FILTER_STRENGTH
    alpha *= (strength / max_strength) if max_strength > 0 else 0.0
    return fmax(0.0, fmin(1.0, alpha))

cdef inline float determine_beta(float boundary_diff, float texture, float strength, int plane_idx) nogil:
    cdef float boundary_flat_thresh, boundary_edge_thresh
    cdef float texture_flat_thresh, texture_textured_thresh
    cdef float beta_flat, beta_textured
    if plane_idx == 0:
        boundary_flat_thresh = Y_BOUNDARY_DIFF_THRESHOLD_FLAT
        boundary_edge_thresh = Y_BOUNDARY_DIFF_THRESHOLD_EDGE
        texture_flat_thresh = Y_TEXTURE_THRESHOLD_FLAT
        texture_textured_thresh = Y_TEXTURE_THRESHOLD_TEXTURED
        beta_flat = Y_BETA_FLAT
        beta_textured = Y_BETA_TEXTURED
    else:
        boundary_flat_thresh = UV_BOUNDARY_DIFF_THRESHOLD_FLAT
        boundary_edge_thresh = UV_BOUNDARY_DIFF_THRESHOLD_EDGE
        texture_flat_thresh = UV_TEXTURE_THRESHOLD_FLAT
        texture_textured_thresh = UV_TEXTURE_THRESHOLD_TEXTURED
        beta_flat = UV_BETA_FLAT
        beta_textured = UV_BETA_TEXTURED
    cdef float texture_ratio = fmax(0.0, fmin(1.0, (texture - texture_flat_thresh) / (texture_textured_thresh - texture_flat_thresh)))
    cdef float beta = beta_flat + (beta_textured - beta_flat) * (1.0 - texture_ratio)
    cdef float max_strength = Y_MAX_FILTER_STRENGTH if plane_idx == 0 else UV_MAX_FILTER_STRENGTH
    beta *= (strength / max_strength) if max_strength > 0 else 0.0
    return fmax(0.0, fmin(1.0, beta))

cdef inline float calculate_boundary_diff(unsigned short[:, :, :] image, int r, int c, int plane, bint is_horizontal) nogil:
    cdef int height = image.shape[0]
    cdef int width = image.shape[1]
    cdef float diff = 0.0
    cdef int i
    if is_horizontal:
        if c == 0: return 0.0
        for i in range(-2, 3):
            if r + i >= 0 and r + i < height:
                diff += fabs(image[r + i, c, plane] - image[r + i, c - 1, plane])
    else:
        if r == 0: return 0.0
        for i in range(-2, 3):
            if c + i >= 0 and c + i < width:
                diff += fabs(image[r, c + i, plane] - image[r - 1, c + i, plane])
    return diff

cdef inline float calculate_local_texture(unsigned short[:, :, :] image, int r, int c, int plane, bint is_horizontal) nogil:
    cdef int height = image.shape[0]
    cdef int width = image.shape[1]
    cdef float texture = 0.0
    cdef int i, j
    cdef int window_size = 5
    if is_horizontal:
        if c == 0: return 0.0
        for i in range(-window_size // 2, window_size // 2 + 1):
            for j in range(-window_size // 2, window_size // 2 + 1):
                 if r + i >= 0 and r + i < height and c + j >= 0 and c + j < width - 1:
                     texture += fabs(image[r + i, c + j + 1, plane] - image[r + i, c + j, plane])
    else:
        if r == 0: return 0.0
        for i in range(-window_size // 2, window_size // 2 + 1):
            for j in range(-window_size // 2, window_size // 2 + 1):
                 if r + i >= 0 and r + i < height - 1 and c + j >= 0 and c + j < width:
                     texture += fabs(image[r + i + 1, c + j, plane] - image[r + i, c + j, plane])
    return texture

cdef inline void apply_filter_2tap(unsigned short[:, :, :] image, int r, int c, int plane, bint is_horizontal, float strength, float alpha, float beta) nogil:
    cdef int p0, q0
    cdef int delta
    cdef int filtered_p0, filtered_q0
    if is_horizontal:
        p0 = image[r, c - 1, plane]
        q0 = image[r, c, plane]
    else:
        p0 = image[r - 1, c, plane]
        q0 = image[r, c, plane]
    delta = <int>(round((q0 - p0) * alpha))
    delta = <int>(fmax(-strength, fmin(strength, delta)))
    filtered_p0 = p0 + delta
    filtered_q0 = q0 - delta
    filtered_p0 = <int>(fmax(0.0, fmin(65535.0, filtered_p0)))
    filtered_q0 = <int>(fmax(0.0, fmin(65535.0, filtered_q0)))
    if is_horizontal:
        image[r, c - 1, plane] = filtered_p0
        image[r, c, plane] = filtered_q0
    else:
        image[r - 1, c, plane] = filtered_p0
        image[r, c, plane] = filtered_q0

cdef inline void apply_filter_4tap(unsigned short[:, :, :] image, int r, int c, int plane, bint is_horizontal, float strength, float alpha, float beta) nogil:
    cdef int p1, p0, q0, q1
    cdef int delta
    cdef int filtered_p1, filtered_p0, filtered_q0, filtered_q1
    if is_horizontal:
        if c < 2 or c >= image.shape[1] - 1: return
        p1 = image[r, c - 2, plane]
        p0 = image[r, c - 1, plane]
        q0 = image[r, c, plane]
        q1 = image[r, c + 1, plane]
    else:
        if r < 2 or r >= image.shape[0] - 1: return
        p1 = image[r - 2, c, plane]
        p0 = image[r - 1, c, plane]
        q0 = image[r, c, plane]
        q1 = image[r + 1, c, plane]
    delta = <int>(round((q0 - p0) * alpha))
    delta = <int>(fmax(-strength, fmin(strength, delta)))
    cdef float beta_factor = 1.0 - beta
    filtered_p0 = p0 + delta
    filtered_q0 = q0 - delta
    filtered_p1 = p1 + <int>(round(delta * beta_factor))
    filtered_q1 = q1 - <int>(round(delta * beta_factor))
    filtered_p1 = <int>(fmax(0.0, fmin(65535.0, filtered_p1)))
    filtered_p0 = <int>(fmax(0.0, fmin(65535.0, filtered_p0)))
    filtered_q0 = <int>(fmax(0.0, fmin(65535.0, filtered_q0)))
    filtered_q1 = <int>(fmax(0.0, fmin(65535.0, filtered_q1)))
    if is_horizontal:
        image[r, c - 2, plane] = filtered_p1
        image[r, c - 1, plane] = filtered_p0
        image[r, c, plane] = filtered_q0
        image[r, c + 1, plane] = filtered_q1
    else:
        image[r - 2, c, plane] = filtered_p1
        image[r - 1, c, plane] = filtered_p0
        image[r, c, plane] = filtered_q0
        image[r + 1, c, plane] = filtered_q1

cdef inline void apply_filter_6tap(unsigned short[:, :, :] image, int r, int c, int plane, bint is_horizontal, float strength, float alpha, float beta) nogil:
    cdef int p2, p1, p0, q0, q1, q2
    cdef int delta
    cdef int filtered_p2, filtered_p1, filtered_p0, filtered_q0, filtered_q1, filtered_q2

    if is_horizontal:
        if c < 3 or c >= image.shape[1] - 2: return
        p2 = image[r, c - 3, plane]
        p1 = image[r, c - 2, plane]
        p0 = image[r, c - 1, plane]
        q0 = image[r, c, plane]
        q1 = image[r, c + 1, plane]
        q2 = image[r, c + 2, plane]
    else:
        if r < 3 or r >= image.shape[0] - 2: return
        p2 = image[r - 3, c, plane]
        p1 = image[r - 2, c, plane]
        p0 = image[r - 1, c, plane]
        q0 = image[r, c, plane]
        q1 = image[r + 1, c, plane]
        q2 = image[r + 2, c, plane]
    delta = <int>(round((q0 - p0) * alpha))
    delta = <int>(fmax(-strength, fmin(strength, delta)))
    cdef float beta_factor1 = 1.0 - beta * 0.5
    cdef float beta_factor2 = 1.0 - beta
    filtered_p0 = p0 + delta
    filtered_q0 = q0 - delta
    filtered_p1 = p1 + <int>(round(delta * beta_factor1))
    filtered_q1 = q1 - <int>(round(delta * beta_factor1))
    filtered_p2 = p2 + <int>(round(delta * beta_factor2))
    filtered_q2 = q2 - <int>(round(delta * beta_factor2))
    filtered_p2 = <int>(fmax(0.0, fmin(65535.0, filtered_p2)))
    filtered_p1 = <int>(fmax(0.0, fmin(65535.0, filtered_p1)))
    filtered_p0 = <int>(fmax(0.0, fmin(65535.0, filtered_p0)))
    filtered_q0 = <int>(fmax(0.0, fmin(65535.0, filtered_q0)))
    filtered_q1 = <int>(fmax(0.0, fmin(65535.0, filtered_q1)))
    filtered_q2 = <int>(fmax(0.0, fmin(65535.0, filtered_q2)))
    if is_horizontal:
        image[r, c - 3, plane] = filtered_p2
        image[r, c - 2, plane] = filtered_p1
        image[r, c - 1, plane] = filtered_p0
        image[r, c, plane] = filtered_q0
        image[r, c + 1, plane] = filtered_q1
        image[r, c + 2, plane] = filtered_q2
    else:
        image[r - 3, c, plane] = filtered_p2
        image[r - 2, c, plane] = filtered_p1
        image[r - 1, c, plane] = filtered_p0
        image[r, c, plane] = filtered_q0
        image[r + 1, c, plane] = filtered_q1
        image[r + 2, c, plane] = filtered_q2

cdef inline void apply_filter_8tap(unsigned short[:, :, :] image, int r, int c, int plane, bint is_horizontal, float strength, float alpha, float beta) nogil:
    cdef int p3, p2, p1, p0, q0, q1, q2, q3
    cdef int delta
    cdef int filtered_p3, filtered_p2, filtered_p1, filtered_p0, filtered_q0, filtered_q1, filtered_q2, filtered_q3
    if is_horizontal:
        if c < 4 or c >= image.shape[1] - 3: return
        p3 = image[r, c - 4, plane]
        p2 = image[r, c - 3, plane]
        p1 = image[r, c - 2, plane]
        p0 = image[r, c - 1, plane]
        q0 = image[r, c, plane]
        q1 = image[r, c + 1, plane]
        q2 = image[r, c + 2, plane]
        q3 = image[r, c + 3, plane]
    else:
        if r < 4 or r >= image.shape[0] - 3: return
        p3 = image[r - 4, c, plane]
        p2 = image[r - 3, c, plane]
        p1 = image[r - 2, c, plane]
        p0 = image[r - 1, c, plane]
        q0 = image[r, c, plane]
        q1 = image[r + 1, c, plane]
        q2 = image[r + 2, c, plane]
        q3 = image[r + 3, c, plane]
    delta = <int>(round((q0 - p0) * alpha))
    delta = <int>(fmax(-strength, fmin(strength, delta)))
    cdef float beta_factor1 = 1.0 - beta * 0.75
    cdef float beta_factor2 = 1.0 - beta * 0.5
    cdef float beta_factor3 = 1.0 - beta * 0.25
    filtered_p0 = p0 + delta
    filtered_q0 = q0 - delta
    filtered_p1 = p1 + <int>(round(delta * beta_factor1))
    filtered_q1 = q1 - <int>(round(delta * beta_factor1))
    filtered_p2 = p2 + <int>(round(delta * beta_factor2))
    filtered_q2 = q2 - <int>(round(delta * beta_factor2))
    filtered_p3 = p3 + <int>(round(delta * beta_factor3))
    filtered_q3 = q3 - <int>(round(delta * beta_factor3))
    filtered_p3 = <int>(fmax(0.0, fmin(65535.0, filtered_p3)))
    filtered_p2 = <int>(fmax(0.0, fmin(65535.0, filtered_p2)))
    filtered_p1 = <int>(fmax(0.0, fmin(65535.0, filtered_p1)))
    filtered_p0 = <int>(fmax(0.0, fmin(65535.0, filtered_p0)))
    filtered_q0 = <int>(fmax(0.0, fmin(65535.0, filtered_q0)))
    filtered_q1 = <int>(fmax(0.0, fmin(65535.0, filtered_q1)))
    filtered_q2 = <int>(fmax(0.0, fmin(65535.0, filtered_q2)))
    filtered_q3 = <int>(fmax(0.0, fmin(65535.0, filtered_q3)))
    if is_horizontal:
        image[r, c - 4, plane] = filtered_p3
        image[r, c - 3, plane] = filtered_p2
        image[r, c - 2, plane] = filtered_p1
        image[r, c - 1, plane] = filtered_p0
        image[r, c, plane] = filtered_q0
        image[r, c + 1, plane] = filtered_q1
        image[r, c + 2, plane] = filtered_q2
        image[r, c + 3, plane] = filtered_q3
    else:
        image[r - 4, c, plane] = filtered_p3
        image[r - 3, c, plane] = filtered_p2
        image[r - 2, c, plane] = filtered_p1
        image[r - 1, c, plane] = filtered_p0
        image[r, c, plane] = filtered_q0
        image[r + 1, c, plane] = filtered_q1
        image[r + 2, c, plane] = filtered_q2
        image[r + 3, c, plane] = filtered_q3

cdef inline void deblock_image(unsigned short[:, :, :] image_in, unsigned short[:, :, :] image_out) nogil:
    cdef int height = image_in.shape[0]
    cdef int width = image_in.shape[1]
    cdef int planes = image_in.shape[2]
    cdef int r, c, plane
    cdef float boundary_diff, texture
    cdef int tap_count
    cdef float strength, alpha, beta
    for plane in prange(planes, nogil=True):
        for r in prange(height):
            for c in prange(width):
                image_out[r, c, plane] = image_in[r, c, plane]
    for plane in prange(planes, nogil=True):
        for r in prange(height):
            for c in prange(BLOCK_SIZE, width, BLOCK_SIZE):
                if r < BORDER_SIZE or r >= height - BORDER_SIZE: continue
                boundary_diff = calculate_boundary_diff(image_in, r, c, plane, True)
                texture = calculate_local_texture(image_in, r, c, plane, True)
                tap_count = determine_tap_count(boundary_diff, texture, plane)
                strength = determine_filter_strength(boundary_diff, texture, plane)
                alpha = determine_alpha(boundary_diff, texture, strength, plane)
                beta = determine_beta(boundary_diff, texture, strength, plane)
                if tap_count == 2: apply_filter_2tap(image_out, r, c, plane, True, strength, alpha, beta)
                elif tap_count == 4: apply_filter_4tap(image_out, r, c, plane, True, strength, alpha, beta)
                elif tap_count == 6: apply_filter_6tap(image_out, r, c, plane, True, strength, alpha, beta)
                elif tap_count == 8: apply_filter_8tap(image_out, r, c, plane, True, strength, alpha, beta)
    for plane in prange(planes, nogil=True):
        for r in prange(BLOCK_SIZE, height, BLOCK_SIZE):
            for c in prange(width):
                if c < BORDER_SIZE or c >= width - BORDER_SIZE: continue
                boundary_diff = calculate_boundary_diff(image_in, r, c, plane, False)
                texture = calculate_local_texture(image_in, r, c, plane, False)
                tap_count = determine_tap_count(boundary_diff, texture, plane)
                strength = determine_filter_strength(boundary_diff, texture, plane)
                alpha = determine_alpha(boundary_diff, texture, strength, plane)
                beta = determine_beta(boundary_diff, texture, strength, plane)
                if tap_count == 2: apply_filter_2tap(image_out, r, c, plane, False, strength, alpha, beta)
                elif tap_count == 4: apply_filter_4tap(image_out, r, c, plane, False, strength, alpha, beta)
                elif tap_count == 6: apply_filter_6tap(image_out, r, c, plane, False, strength, alpha, beta)
                elif tap_count == 8: apply_filter_8tap(image_out, r, c, plane, False, strength, alpha, beta)
