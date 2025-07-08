#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device = 0;  // You can change this if you have multiple GPUs
    cudaDeviceProp prop;

    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Device " << device << " name: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total global memory: " << (prop.totalGlobalMem >> 20) << " MB" << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max block dimensions: ["
              << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", "
              << prop.maxThreadsDim[2] << "]" << std::endl;
    std::cout << "Max grid dimensions: ["
              << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", "
              << prop.maxGridSize[2] << "]" << std::endl;

    return 0;
}
