import vapoursynth as vs
from vapoursynth import core
import numpy as np

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
    