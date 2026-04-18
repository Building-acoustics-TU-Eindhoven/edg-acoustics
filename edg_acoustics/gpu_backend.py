"""GPU backend for edg-acoustics using CuPy."""
import numpy
try:
    import cupy
    HAS_GPU = True
except ImportError:
    cupy = None
    HAS_GPU = False

_use_gpu = False
xp = numpy

def enable_gpu():
    global _use_gpu, xp
    if not HAS_GPU:
        return False
    _use_gpu = True
    xp = cupy
    return True

def disable_gpu():
    global _use_gpu, xp
    _use_gpu = False
    xp = numpy

def is_gpu():
    return _use_gpu

def to_device(arr):
    if _use_gpu and isinstance(arr, numpy.ndarray):
        return cupy.asarray(arr)
    return arr

def to_host(arr):
    if _use_gpu and HAS_GPU and isinstance(arr, cupy.ndarray):
        return cupy.asnumpy(arr)
    return arr

def sync():
    if _use_gpu and HAS_GPU:
        cupy.cuda.Stream.null.synchronize()
