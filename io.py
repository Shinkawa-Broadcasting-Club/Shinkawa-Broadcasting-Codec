from zstd import ZSTD_compress, ZSTD_uncompress
import numpy as np
# get header
def get_source_header(clip):
	None

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