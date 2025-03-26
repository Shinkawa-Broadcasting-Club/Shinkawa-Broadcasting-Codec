# sbc internal libraries
from sbc.quantization import quantize, dequantize, gen_aq, css
from sbc.io import entropy_coding_fwd, entropy_coding_bwd
from sbc.transform import dct_3d_fwd, dct_3d_bwd
from sbc.vsnpconv import vs_to_np
from sbc.mt import mt_run
# external libraries
from threading import Thread, Semaphore
from colour import YCbCr_to_RGB
from itertools import product
import vapoursynth as vs
import numpy as np
import cv2
core = vs.core
# settings
threads = 4
q = 4
preset = 22
path = r"C:\Users\yyuuk\35_N71Tドラ_シャッター.mp4"
transfer = "709"
aq_strength = 3
def sbc_encode(path, q, aq_strength, preset, threads, transfer):
	# restrict threads to boost encoding speed
	semaphore = Semaphore(threads)
	# video loading
	clip = core.lsmas.LWLibavSource(path, format="YUV444P16")
	video = vs_to_np(clip[1000:1008])
	# padding
	x = video.shape
	y = [(8 - (8 % x[a])) % 8 for a in range(3)]
	plane = np.pad(video, ((0, y[0]), (0, y[1]), (0, y[2]), (0, 0)))
	del video
	plane[x[0]:-1] = plane[x[0] - 1:x[0]]
	plane[:, x[1]:-1] = plane[:, x[1] - 1:x[1]]
	plane[:, :, x[2]:-1] = plane[:, :, x[2] - 1:x[2]]
	del x, y
	# 3D-DCT
	coef = dct_3d_fwd(plane.astype(np.float32))
	coef[::8, ::8, ::8] -= 16776960
	coef = (coef / 524280 * 2047).astype(np.float16)
	del plane
	# chroma subsampling
	thread = [Thread(target = css, args = (semaphore, coef, i, j, k)) for i, j, k in product(range(8), range(8), range(3))]
	mt_run(thread)
	del thread
	# quantize coeffecient
	palette = np.zeros((2 ** (2 * q - 1), 8, 8, 8, 6))
	c = coef.shape
	quantized = np.zeros((8, c[1], c[2]))
	aq = gen_aq(q, aq_strength)
	stack_multiple = np.array([[3, 6, 6, 6, 2, 2, 2, 2], [0, 3, 6, 6, 2, 2, 2, 2], [0, 0, 3, 6, 2, 2, 2, 2], [0, 0, 0, 3, 2, 2, 2, 2], [0, 0, 0, 0, 1, 2, 2, 2], [0, 0, 0, 0, 0, 1, 2, 2], [0, 0, 0, 0, 0, 0, 1, 2], [0, 0, 0, 0, 0, 0, 0, 1]])
	aq *= np.round(stack_multiple)
	coef = coef.transpose((1, 2, 3, 0))
	coef = np.dstack((coef, coef)).transpose((3, 0, 1, 2))
	for i, j in product(range(8), repeat = 2):
		coef[:, i::8, j::8, 3:] = coef[:, 7 - i::8, 7 - j::8, :3]
	thread = [Thread(target = quantize, args = (semaphore, coef, aq, palette, quantized, g, h, i, 6)) for g, h, i in product(range(8), repeat = 3)]
	mt_run(thread)
	del thread, coef
	codebook = palette.astype(np.float16)
	del palette
	# pack array
	data = quantized.copy()
	del quantized
	data = data.astype(np.uint8) if q < 5 else data.astype(np.uint16)
	for g, h, i in product(range(4), range(8), range(8)):
		data[g::8, h::8, i::8] = data[g::8, h::8, i::8] * (1 << aq[7 - g, 7 - h, 7 - i]) + data[7 - g::8, 7 - h::8, 7 - i::8]
	# stacking
	data[2::8] = data[1::8]
	data[4::8] = data[2::8]
	data[6::8] = data[3::8]
	data = data[::2]
	s = data.shape
	# entropy coding
	compressed = entropy_coding_fwd(data, preset, threads, s)
	return compressed + b'EOB', codebook, aq

result, codebook = sbc_encode(path, q, aq_strength, preset, threads, transfer)

splited = marked.split(b'EOB')
data = entropy_coding_bwd(compressed, s)
# unpack array
data = np.vstack((data, data))
aq = gen_aq(q, aq_strength).astype(int)
for g, h, i in product(range(4), range(8), range(8)):
	n = (2 ** aq[7 - g, 7 - h, 7 - i].astype(np.uint16))
	data[g::8, h::8, i::8] = data[g::8, h::8, i::8] // n
	data[g + 4::8, h::8, i::8] = data[g + 4::8, h::8, i::8] - data[g::8, h::8, i::8] * n

quantized = data.copy()
del data
# dequantize coeffecient
s = quantized.shape
coef = np.zeros((8, s[1], s[2], 3)).astype(np.float16)
thread = [Thread(target = dequantize, args = (coef, quantized, codebook, h, i, j)) for h, i, j in product(range(8), repeat = 3)]
mt_run(thread)
del quantized, thread
# 3D-IDCT
coef = dct_3d_bwd(coef.astype(np.float32) * 1048560 / 2047)
# normalize coefficient and convert to 16-bit integer
prev = YCbCr_to_RGB(np.clip(coef // 8, a_min = 0, a_max = 65535).astype(np.uint16)[:, :clip.height, :clip.width], in_bits = 16).astype(np.uint16)
del coef
fps = int(1000 / clip.fps.numerator * clip.fps.denominator)
for e in range(len(prev)):
	cv2.imshow('image window', prev[e])
	cv2.waitKey(fps)

cv2.destroyAllWindows()
