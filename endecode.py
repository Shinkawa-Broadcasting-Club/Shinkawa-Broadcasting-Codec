# sbc internal libraries
from sbc.quantization import quantize, dequantize, gen_aq, css
from sbc.sbc_io import entropy_coding_fwd, get_source_header, get_binary_header
from sbc.transform import dct_3d_fwd, dct_3d_bwd
from sbc.vsnpconv import vs_to_np
from sbc.mt import mt_run
# external libraries
from threading import Thread, Semaphore
from zstd import ZSTD_uncompress
from colour import YCbCr_to_RGB
from itertools import product
import vapoursynth as vs
import numpy as np
import cv2
import os
core = vs.core
# settings
threads = 4
q = 4
preset = 22
path = r"C:\Users\yyuuk\35_N71Tドラ_シャッター.mp4"
output = r"C:\Users\yyuuk\35_N71Tドラ_シャッター.sbc"
transfer = "709"
aq_strength = 3

def sbc_encode(clip, q, preset, threads, transfer):
	# restrict threads to boost encoding speed
	semaphore = Semaphore(threads)
	# video loading
	video = (vs_to_np(clip) / 1048560 * 2047).astype(np.float16)
	# padding
	x = video.shape
	y = [np.uint8(np.ceil(x[a] / 8) * 8 - x[a]) for a in range(3)]
	plane = np.pad(video, ((0, y[0]), (0, y[1]), (0, y[2]), (0, 0)))
	del video
	plane[x[0]:-1] = plane[x[0] - 1:x[0]]
	plane[:, x[1]:-1] = plane[:, x[1] - 1:x[1]]
	plane[:, :, x[2]:-1] = plane[:, :, x[2] - 1:x[2]]
	del x, y
	# 3D-DCT
	coef = dct_3d_fwd(plane.astype(np.float32)).astype(np.float16)
	del plane
	# chroma subsampling
	thread = [Thread(target = css, args = (semaphore, coef, i, j, k)) for i, j, k in product(range(8), range(8), range(3))]
	mt_run(thread)
	del thread
	# quantize coeffecient
	c = coef.shape
	quantized = np.zeros((8, c[1], c[2]))
	aq = np.round(gen_aq(q)).astype(np.int8)
	palette = np.zeros((2 ** (2 * q - 1), 8, 8, 8, 3))
	thread = [Thread(target = quantize, args = (semaphore, coef, aq, palette, quantized, g, h, i)) for g, h, i in product(range(8), repeat = 3)]
	mt_run(thread)
	del thread, coef
	codebook = np.array([0])
	for a, b, c, d in product(range(8), range(8), range(8), range(3)):
		if b < 4 or c < 4 or d == 0:
			codebook = np.concatenate([codebook, palette[:2 ** aq[a, b, c], a, b, c, d]])
	codebook = codebook[1:].astype(np.float16)
	del palette
	# pack array
	data = quantized.copy()
	del quantized
	data = data.astype(np.uint16)
	for g, h, i in product(range(4), range(8), range(8)):
		data[g::8, h::8, i::8] = data[g::8, h::8, i::8] * (2 ** aq[7 - g, 7 - h, 7 - i]) + data[7 - g::8, 7 - h::8, 7 - i::8]
	# stacking
	data[2::8] = data[1::8]
	data[4::8] = data[2::8]
	data[6::8] = data[3::8]
	data = data[::2]
	s = data.shape
	# entropy coding
	compressed = entropy_coding_fwd(data, codebook, preset, threads, s)
	return compressed + b'EOB'

def sbc_encoder(path, q, output, transfer):
	if not(0 < q < 8):
		return ("q must be 0 < q < 8 integer")
	clip = core.lsmas.LWLibavSource(path, format="YUV444P16")
	header = get_source_header(clip)
	aq = np.uint8(q).tobytes()
	with open(output, "wb") as f:
		f.write(header)
		f.write(aq)
	f = open(output, "ab")
	for n in range(np.ceil(clip.num_frames / 8).astype(np.uint32)):
		r = sbc_encode(clip[n:n + 8], q, preset, threads, transfer)
		f.write(r)
	f.close()

def sbc_decode(splited, s, q):
	data = ZSTD_uncompress(splited)
	# unpack array
	aq = np.round(gen_aq(q)).astype(np.int8)
	codebook = np.zeros((1, 8, 8, 8, 3))
	for n in range(2 ** (2 * q.astype(np.uint64) - 1) - 1):
		codebook = np.vstack((codebook, np.zeros((1, 8, 8, 8, 3))))
	for a, b, c, d in product(range(8), range(8), range(8), range(3)):
		if b < 4 or c < 4 or d == 0:
			codebook[:2 ** aq[a, b, c], a, b ,c, d] = np.frombuffer(data[:2 ** (aq[a, b, c].astype(np.uint64) + 1)], np.float16)
			data = data[2 ** (aq[a, b, c].astype(np.uint64) + 1):]
	data = np.frombuffer(data, dtype = np.uint16)
	data = np.split(data, s[1])
	data = np.stack(data, axis=-1)
	data = np.split(data, s[2])
	data = np.stack(data, axis=-1)
	data = np.vstack((data, data))
	for g, h, i in product(range(4), range(8), range(8)):
		n = (2 ** aq[7 - g, 7 - h, 7 - i].astype(np.uint16))
		data[g::8, h::8, i::8] = data[g::8, h::8, i::8] // n
		data[g + 4::8, h::8, i::8] = data[g + 4::8, h::8, i::8] - data[g::8, h::8, i::8] * n
	quantized = data.copy()
	del data
	# dequantize coeffecient
	coef = np.zeros((8, s[1], s[2], 3)).astype(np.float16)
	thread = [Thread(target = dequantize, args = (coef, quantized, codebook, h, i, j)) for h, i, j in product(range(8), repeat = 3)]
	mt_run(thread)
	del quantized, thread
	# 3D-IDCT
	coef = dct_3d_bwd(coef.astype(np.float32) * 1048560 / 2047)
	return np.clip(coef // 8, a_min = 0, a_max = 65535).astype(np.uint16)[:, :s[1], :s[2]]

def sbc_decoder(path, play):
	with open(path, "rb") as f:
		index = get_binary_header(f.read(12))
		s = np.array([4, index[1], index[0]])
		q = np.frombuffer(f.read(1), np.int8)[0]
		splited = list(filter(lambda x: x != b'', f.read().split(b'EOB')))
		for c in range(np.ceil(index[2] / 8).astype(np.uint32)):
			r = sbc_decode(splited[c], s, q)
			
	return 

#sbc_encoder(path, q, output, transfer)
sbc_decoder(output, 0)

def play(coef, clip):
	# normalize coefficient and convert to 16-bit integer
	prev = YCbCr_to_RGB(np.clip(coef // 8, a_min = 0, a_max = 65535).astype(np.uint16)[:, :clip.height, :clip.width], in_bits = 16).astype(np.uint16)
	del coef
	fps = int(1000 / clip.fps.numerator * clip.fps.denominator)
	for e in range(len(prev)):
		cv2.imshow('image window', prev[e])
		cv2.waitKey(fps)
	cv2.destroyAllWindows()
