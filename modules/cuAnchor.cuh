#include <cuda_runtime.h>


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

