from itertools import product
import numpy as np
def median_cut(img_array, color_count):
	img_shape = img_array.shape
	flat_img_array = img_array.reshape(-1, 3)
	cubes = [(flat_img_array, list(range(flat_img_array.shape[0])))]
	while len(cubes) < color_count:
		cubes = sorted(cubes, key=lambda cube: float(np.max(np.ptp(cube[0], axis=0))) if cube[0].size > 0 else 0, reverse=True)
		largest_cube, indices = cubes.pop(0)
		axis = np.argmax(np.ptp(largest_cube, axis=0))
		sorted_order = np.argsort(largest_cube[:, axis])
		largest_cube = largest_cube[sorted_order]
		sorted_pixel_indices = [indices[i] for i in sorted_order]
		median_index = len(largest_cube) // 2
		cubes.append((largest_cube[:median_index], sorted_pixel_indices[:median_index]))
		cubes.append((largest_cube[median_index:], sorted_pixel_indices[median_index:]))
	palette = [np.mean(cube[0], axis=0).astype(np.float64) for cube in cubes]
	quantized_indices = np.empty(flat_img_array.shape[0], dtype=int)
	for cube_idx, (_, indices) in enumerate(cubes):
		for pixel_index in indices:
			quantized_indices[pixel_index] = cube_idx
	indexed_image = quantized_indices.reshape(img_shape[0], img_shape[1])
	return palette, indexed_image

def gen_aq(q, depth):
	if depth != depth // 1 or q != q // 1:
		print("q and depth must be integer")
		return
	if q < depth + 1:
		print("q must be => depth + 1")
		return
	aq = np.zeros((8, 8, 8))
	if depth > 0:
		aq[:4, :4, :4] += 1
		aq[4:, :4, :4] += 1
		aq[:4, 4:, :4] += 1
		aq[:4, :4, 4:] += 1
		aq[:4, 4:, 4:] -= 1
		aq[4:, :4, 4:] -= 1
		aq[4:, 4:, :4] -= 1
		aq[4:, 4:, 4:] -= 1
	if depth > 1:
		for a, b, c in product(range(2), repeat = 3):
			aq[a::4, b::4, c::4] += 1
			aq[3 - a::4, b::4, c::4] += 1
			aq[a::4, 3 - b::4, c::4] += 1
			aq[a::4, b::4, 3 - c::4] += 1
			aq[a::4, 3 - b::4, 3 - c::4] -= 1
			aq[3 - a::4, b::4, 3 - c::4] -= 1
			aq[3 - a::4, 3 - b::4, c::4] -= 1
			aq[3 - a::4, 3 - b::4, 3 - c::4] -= 1
	if depth > 2:
		aq[::2, ::2, ::2] += 1
		aq[1::2, ::2, ::2] += 1
		aq[::2, 1::2, ::2] += 1
		aq[::2, ::2, 1::2] += 1
		aq[::2, 1::2, 1::2] -= 1
		aq[1::2, ::2, 1::2] -= 1
		aq[1::2, 1::2, ::2] -= 1
		aq[1::2, 1::2, 1::2] -= 1
	return aq + q

def quantize(semaphore, coef, aq, palette, quantized, g, h, i):
	semaphore.acquire()
	palette[:2 ** aq[g, h, i], g, h, i], quantized[g, h::8, i::8] = median_cut(coef[g, h::8, i::8], 2 ** aq[g, h, i])
	semaphore.release()

def css(semaphore, coef, i, j, k):
	semaphore.acquire()
	coef[:, i::8, j::8, k] = 0 if k > 0 and (i > 3 or j > 3) else coef[:, i::8, j::8, k]
	semaphore.release()

def dequantize(semaphore, coef, quantized, codebook, h, i, j, k):
	semaphore.acquire()
	for n in range(3):
		coef[h, i::8, j::8, n] = np.choose([quantized[h, i::8, j::8] == l for l in range(k)], [codebook[m, h, i, j, n] for m in range(k)])
	semaphore.release()