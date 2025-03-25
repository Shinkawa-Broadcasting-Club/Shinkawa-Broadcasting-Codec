import vapoursynth as vs
import numpy as np
core = vs.core

def vs_to_np(i):
    ndarray = []
    planes = []
    for m in range(i.format.num_planes):
        for n in range(i.num_frames):
            ndarray.append(np.array(i.get_frame(n)[m], copy=False))
        planes.append(np.stack(ndarray, axis=0))
        ndarray = []
    result = np.stack(planes, axis=-1)
    del ndarray, planes
    return result

def np_to_vs(numpy_array, num, den):
	num_frames, height, width, channels = numpy_array.shape
	if channels != 3:
		raise ValueError("Input array must have Y, U, V channels")
	def frame_generator(n, frame):
		for plane in range(3):
			plane_data = numpy_array[n, :, :, plane].tobytes()
			frame[plane].replace(plane_data)
	clip = core.std.BlankClip(format=vs.YUV444P16, length=num_frames, width=width, height=height, fpsnum=num, fpsden=den)
	return clip.std.ModifyFrame(clips=clip, selector=frame_generator)