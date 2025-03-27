from zstd import ZSTD_compress, ZSTD_uncompress
import vapoursynth as vs
import numpy as np
core = vs.core
# get header
def get_source_header(clip):
	index = np.array([clip.width, clip.height, clip.num_frames, clip.fps.numerator, clip.fps.denominator]).astype(np.uint32)
	if index[0] > 65535 or index[1] > 65535 or index[2] != clip.num_frames or index[3] > 65535 or index[4] > 65535:
		return
	else:
		index[1] += index[0] << 16
		index[4] += index[3] << 16
		return np.array([index[1], index[2], index[4]]).tobytes()

def get_binary_header(binary):
	index = np.frombuffer(binary[:3].to_bytes(), np.uint32)
	return np.array([index[0] >> 16, np.bitwise_and(index[0], 255), index[1], index[2] >> 16, np.bitwise_and(index[2], 255)])

def entropy_coding_fwd(data, preset, threads, s):
	data = np.dsplit(data, s[2])
	data = np.concatenate(data)[:, :, 0]
	data = np.hsplit(data, s[1])
	data = np.concatenate(data)[:, 0]
	compressed = ZSTD_compress(data, preset, threads)
	return compressed

def entropy_coding_bwd(compressed, s):
	decompressed = ZSTD_uncompress(compressed)
	data = np.frombuffer(decompressed, dtype = np.uint8)
	data = np.split(data, s[1])
	data = np.stack(data, axis=-1)
	data = np.split(data, s[2])
	data = np.stack(data, axis=-1)
	return data