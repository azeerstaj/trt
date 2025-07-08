// C++ wrapper function for anchor generation CUDA kernel
// #include "cuAnchor.cuh"
#include <torch/extension.h>
#include <vector>

__global__ void generate_anchors_kernel(
    const float* base_anchors,     // Shape: (total_base_anchors, 4)
    const int* feature_map_info,   // Shape: (num_levels, 4) -> [feat_h, feat_w, stride_h, stride_w]
    const int* anchor_counts,      // Shape: (num_levels,) -> number of base anchors per level
    const int* level_offsets,      // Shape: (num_levels,) -> cumulative base anchor offsets
    const int* output_offsets,     // Shape: (num_levels,) -> cumulative output anchor offsets
    float* output_anchors,         // Shape: (total_output_anchors, 4)
    int num_levels,
    int total_output_anchors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_output_anchors) return;

    // Find which level this anchor belongs to
    int level = 0;
    for (int i = 0; i < num_levels; i++) {
        if (idx < output_offsets[i]) {
            level = i - 1;
            break;
        }
    }
    if (level < 0) level = num_levels - 1; // Handle last level

    // Get level-specific info
    int feat_h = feature_map_info[level * 4 + 0];
    int feat_w = feature_map_info[level * 4 + 1];
    int stride_h = feature_map_info[level * 4 + 2];
    int stride_w = feature_map_info[level * 4 + 3];
    int num_base_anchors = anchor_counts[level];

    // Calculate position within this level
    int level_start = (level == 0) ? 0 : output_offsets[level - 1];
    int pos_in_level = idx - level_start;

    // Decompose position into spatial location and base anchor index
    int spatial_pos = pos_in_level / num_base_anchors;
    int base_anchor_idx = pos_in_level % num_base_anchors;

    // Convert spatial position to (y, x) coordinates
    int y = spatial_pos / feat_w;
    int x = spatial_pos % feat_w;

    // Calculate shift
    float shift_x = x * stride_w;
    float shift_y = y * stride_h;

    // Get base anchor
    int base_anchor_global_idx = level_offsets[level] + base_anchor_idx;
    float base_x1 = base_anchors[base_anchor_global_idx * 4 + 0];
    float base_y1 = base_anchors[base_anchor_global_idx * 4 + 1];
    float base_x2 = base_anchors[base_anchor_global_idx * 4 + 2];
    float base_y2 = base_anchors[base_anchor_global_idx * 4 + 3];

    // Apply shift and store result
    output_anchors[idx * 4 + 0] = base_x1 + shift_x;
    output_anchors[idx * 4 + 1] = base_y1 + shift_y;
    output_anchors[idx * 4 + 2] = base_x2 + shift_x;
    output_anchors[idx * 4 + 3] = base_y2 + shift_y;
}

// Host wrapper functions
void launch_generate_anchors_kernel(
    const float* base_anchors,
    const int* feature_map_info,
    const int* anchor_counts,
    const int* level_offsets,
    const int* output_offsets,
    float* output_anchors,
    int num_levels,
    int total_output_anchors
) {
    int block_size = 256;
    int grid_size = (total_output_anchors + block_size - 1) / block_size;

    generate_anchors_kernel<<<grid_size, block_size>>>(
        base_anchors, feature_map_info, anchor_counts,
        level_offsets, output_offsets, output_anchors,
        num_levels, total_output_anchors
    );
}

torch::Tensor generate_anchors_cuda(
    torch::Tensor base_anchors,           // Shape: (total_base_anchors, 4)
    torch::Tensor feature_map_info,       // Shape: (num_levels, 4) -> [feat_h, feat_w, stride_h, stride_w]
    torch::Tensor anchor_counts,          // Shape: (num_levels,) -> number of base anchors per level
    torch::Tensor level_offsets,          // Shape: (num_levels,) -> cumulative base anchor offsets
    torch::Tensor output_offsets,         // Shape: (num_levels,) -> cumulative output anchor offsets
    int total_output_anchors
) {
    // Check that tensors are on CUDA
    TORCH_CHECK(base_anchors.device().is_cuda(), "base_anchors must be a CUDA tensor");
    TORCH_CHECK(feature_map_info.device().is_cuda(), "feature_map_info must be a CUDA tensor");
    TORCH_CHECK(anchor_counts.device().is_cuda(), "anchor_counts must be a CUDA tensor");
    TORCH_CHECK(level_offsets.device().is_cuda(), "level_offsets must be a CUDA tensor");
    TORCH_CHECK(output_offsets.device().is_cuda(), "output_offsets must be a CUDA tensor");
    
    // Check tensor types
    TORCH_CHECK(base_anchors.dtype() == torch::kFloat32, "base_anchors must be float32");
    TORCH_CHECK(feature_map_info.dtype() == torch::kInt32, "feature_map_info must be int32");
    TORCH_CHECK(anchor_counts.dtype() == torch::kInt32, "anchor_counts must be int32");
    TORCH_CHECK(level_offsets.dtype() == torch::kInt32, "level_offsets must be int32");
    TORCH_CHECK(output_offsets.dtype() == torch::kInt32, "output_offsets must be int32");
    
    // Check tensor shapes
    TORCH_CHECK(base_anchors.dim() == 2 && base_anchors.size(1) == 4, "base_anchors must be (N, 4)");
    TORCH_CHECK(feature_map_info.dim() == 2 && feature_map_info.size(1) == 4, "feature_map_info must be (num_levels, 4)");
    TORCH_CHECK(anchor_counts.dim() == 1, "anchor_counts must be 1D");
    TORCH_CHECK(level_offsets.dim() == 1, "level_offsets must be 1D");
    TORCH_CHECK(output_offsets.dim() == 1, "output_offsets must be 1D");
    
    int num_levels = feature_map_info.size(0);
    
    // Create output tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(base_anchors.device());
    torch::Tensor output_anchors = torch::empty({total_output_anchors, 4}, options);
    
    // Launch CUDA kernel
    launch_generate_anchors_kernel(
        base_anchors.data_ptr<float>(),
        feature_map_info.data_ptr<int>(),
        anchor_counts.data_ptr<int>(),
        level_offsets.data_ptr<int>(),
        output_offsets.data_ptr<int>(),
        output_anchors.data_ptr<float>(),
        num_levels,
        total_output_anchors
    );
    
    return output_anchors;
}

// // Python binding
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("generate_anchors_cuda", &generate_anchors_cuda, 
//           "Generate anchors using CUDA (multi-level)");
// }
