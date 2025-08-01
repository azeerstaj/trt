import onnx_graphsurgeon as gs
import numpy as np
import onnx
import os
import sys

import tensorrt as trt
from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    TrtRunner,
    create_network,
    engine_from_network,
    save_engine,
    engine_from_path,
)

from polygraphy import mod

cuda = mod.lazy_import("cuda.cuda")
cudart = mod.lazy_import("cuda.cudart")
nvrtc = mod.lazy_import("cuda.nvrtc")

torch = mod.lazy_import("torch")
torch.manual_seed(0)
cp = mod.lazy_import("cupy")

def volume(d):
    return np.prod(d)

# Taken from https://github.com/NVIDIA/cuda-python/blob/main/examples/common/helper_cuda.py #
def checkCudaErrors(result):
    def _cudaGetErrorEnum(error):
        if isinstance(error, cuda.CUresult):
            err, name = cuda.cuGetErrorName(error)
            return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
        elif isinstance(error, cudart.cudaError_t):
            return cudart.cudaGetErrorName(error)[1]
        elif isinstance(error, nvrtc.nvrtcResult):
            return nvrtc.nvrtcGetErrorString(error)[1]
        else:
            raise RuntimeError("Unknown error type: {}".format(error))

    if result[0].value:
        raise RuntimeError(
            "CUDA error code={}({})".format(
                result[0].value, _cudaGetErrorEnum(result[0])
            )
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


# Taken from https://github.com/NVIDIA/cuda-python/blob/main/examples/common/common.py
class KernelHelper:
    def __init__(self, code, devID):
        prog = checkCudaErrors(
            nvrtc.nvrtcCreateProgram(str.encode(code), b"sourceCode.cu", 0, [], [])
        )
        CUDA_HOME = os.getenv("CUDA_HOME")
        if CUDA_HOME == None:
            CUDA_HOME = os.getenv("CUDA_PATH")
        if CUDA_HOME == None:
            raise RuntimeError("Environment variable CUDA_HOME or CUDA_PATH is not set")
        include_dirs = os.path.join(CUDA_HOME, "include")

        # Initialize CUDA
        checkCudaErrors(cudart.cudaFree(0))

        major = checkCudaErrors(
            cudart.cudaDeviceGetAttribute(
                cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, devID
            )
        )
        minor = checkCudaErrors(
            cudart.cudaDeviceGetAttribute(
                cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, devID
            )
        )
        _, nvrtc_minor = checkCudaErrors(nvrtc.nvrtcVersion())
        use_cubin = nvrtc_minor >= 1
        prefix = "sm" if use_cubin else "compute"
        arch_arg = bytes(f"--gpu-architecture={prefix}_{major}{minor}", "ascii")

        try:
            opts = [
                b"--fmad=true",
                arch_arg,
                "--include-path={}".format(include_dirs).encode("UTF-8"),
                b"--std=c++11",
                b"-default-device",
            ]
            checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        except RuntimeError as err:
            logSize = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * logSize
            checkCudaErrors(nvrtc.nvrtcGetProgramLog(prog, log))
            print(log.decode())
            print(err)
            exit(-1)

        if use_cubin:
            dataSize = checkCudaErrors(nvrtc.nvrtcGetCUBINSize(prog))
            data = b" " * dataSize
            checkCudaErrors(nvrtc.nvrtcGetCUBIN(prog, data))
        else:
            dataSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
            data = b" " * dataSize
            checkCudaErrors(nvrtc.nvrtcGetPTX(prog, data))

        self.module = checkCudaErrors(cuda.cuModuleLoadData(np.char.array(data)))

    def getFunction(self, name):
        return checkCudaErrors(cuda.cuModuleGetFunction(self.module, name))


class CudaCtxManager(trt.IPluginResource):
    def __init__(self, device=None):
        trt.IPluginResource.__init__(self)
        self.device = device
        self.cuda_ctx = None

    def clone(self):
        cloned = CudaCtxManager()
        cloned.__dict__.update(self.__dict__)
        # Delay the CUDA ctx creation until clone()
        # since only a cloned resource is registered by TRT
        _, cloned.cuda_ctx = cuda.cuCtxCreate(0, self.device)
        return cloned

    def release(self):
        checkCudaErrors(cuda.cuCtxDestroy(self.cuda_ctx))

class UnownedMemory:
    def __init__(self, ptr, shape, dtype):
        mem = cp.cuda.UnownedMemory(ptr, volume(shape) * cp.dtype(dtype).itemsize, self)
        cupy_ptr = cp.cuda.MemoryPointer(mem, 0)
        self.d = cp.ndarray(shape, dtype=dtype, memptr=cupy_ptr)