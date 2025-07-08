// C++ wrapper function for anchor generation CUDA kernel
// #include "cuAnchor.cuh"
#include <torch/extension.h>
#include <vector>

__global__ void generate_anchors_single_level_kernel(
    const float* base_anchors,    // Shape: (num_base_anchors, 4)
    float* output_anchors,        // Shape: (feat_h * feat_w * num_base_anchors, 4)
    int feat_h, int feat_w,
    int stride_h, int stride_w,
    int num_base_anchors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_anchors = feat_h * feat_w * num_base_anchors;

    if (idx >= total_anchors) return;

    // Decompose index
    int spatial_pos = idx / num_base_anchors;
    int base_anchor_idx = idx % num_base_anchors;

    // Convert to (y, x) coordinates
    int y = spatial_pos / feat_w;
    int x = spatial_pos % feat_w;

    // Calculate shift
    float shift_x = x * stride_w;
    float shift_y = y * stride_h;

    // Get base anchor and apply shift
    float base_x1 = base_anchors[base_anchor_idx * 4 + 0];
    float base_y1 = base_anchors[base_anchor_idx * 4 + 1];
    float base_x2 = base_anchors[base_anchor_idx * 4 + 2];
    float base_y2 = base_anchors[base_anchor_idx * 4 + 3];

    output_anchors[idx * 4 + 0] = base_x1 + shift_x;
    output_anchors[idx * 4 + 1] = base_y1 + shift_y;
    output_anchors[idx * 4 + 2] = base_x2 + shift_x;
    output_anchors[idx * 4 + 3] = base_y2 + shift_y;
}


// Host wrapper functions
void launch_generate_anchors_kernel_single(
    const float* base_anchors,    // Shape: (num_base_anchors, 4)
    float* output_anchors,        // Shape: (feat_h * feat_w * num_base_anchors, 4)
    int feat_h, int feat_w,
    int stride_h, int stride_w,
    int num_base_anchors
) {
    int block_size = 256;
    int grid_size = (feat_h * feat_w * num_base_anchors + block_size - 1) / block_size;

    generate_anchors_single_level_kernel<<<grid_size, block_size>>>(
        base_anchors, output_anchors,
        feat_h, feat_w,
        stride_h, stride_w,
        num_base_anchors
    );
}

torch::Tensor generate_anchors_single_cuda(
    torch::Tensor base_anchors,           // Shape: (total_base_anchors, 4)
    int feat_h, int feat_w,       
    int stride_h, int stride_w,
    int anchor_counts
) {
    // Check that tensors are on CUDA
    TORCH_CHECK(base_anchors.device().is_cuda(), "base_anchors must be a CUDA tensor");
    
    // Check tensor types
    TORCH_CHECK(base_anchors.dtype() == torch::kFloat32, "base_anchors must be float32");
    
    // Check tensor shapes
    TORCH_CHECK(base_anchors.dim() == 2 && base_anchors.size(1) == 4, "base_anchors must be (N, 4)");
    
    // Create output tensor
    auto total_output_anchors = feat_h * feat_w * base_anchors.size(0);
    // std::cout << "Total Output Anchors: " << total_output_anchors;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(base_anchors.device());
    torch::Tensor output_anchors = torch::empty({total_output_anchors, 4}, options);
    
    // Launch CUDA kernel
    launch_generate_anchors_kernel_single(
        base_anchors.data_ptr<float>(),
        output_anchors.data_ptr<float>(),
        feat_h, feat_w,
        stride_h, stride_w,
        anchor_counts
    );
    
    return output_anchors;
}

// Python binding
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("generate_anchors_single_cuda", &generate_anchors_single_cuda, 
//           "Generate anchors using CUDA (single-level)");
// }
