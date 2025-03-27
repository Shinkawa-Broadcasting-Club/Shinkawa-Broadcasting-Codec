from itertools import product
import numpy as np
def median_cut(img_array, color_count):
	img_shape = img_array.shape
	flat_img_array = img_array.reshape(-1, 6)
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
	palette = [np.mean(cube[0], axis=0).astype(np.float16) for cube in cubes]
	quantized_indices = np.empty(flat_img_array.shape[0], dtype=int)
	for cube_idx, (_, indices) in enumerate(cubes):
		for pixel_index in indices:
			quantized_indices[pixel_index] = cube_idx
	indexed_image = quantized_indices.reshape(img_shape[0], img_shape[1])
	return palette, indexed_image

def gen_aq(q, depth):
	if q < depth + 1:
		print("q must be => depth + 1")
		return
	aq = np.zeros((8, 8, 8))
	for g, h, i in product(range(8), repeat = 3):
		aq[g, h ,i] = depth * (2 / 21 * (g + h + i) - 1)
	return aq + q

def quantize(semaphore, coef, aq, palette, quantized, g, h, i):
	semaphore.acquire()
	col = 2 ** aq[g, h, i]
	palette[:col, g, h, i], quantized[g, h::8, i::8] = median_cut(coef[g, h::8, i::8], col)
	semaphore.release()

def css(semaphore, coef, i, j, k):
	semaphore.acquire()
	coef[:, i::8, j::8, k] = 0 if k > 0 and (i > 3 or j > 3) else coef[:, i::8, j::8, k]
	semaphore.release()

def dequantize(coef, quantized, codebook, h, i, j):
	for k, l, n in product(range(coef.shape[1] - 1), range(coef.shape[2] - 1), range(3)):
		coef[h, k * 8 + i, l * 8 + j, n] = codebook[quantized[h, k * 8 + i, l * 8 + j], h, i, j, n]