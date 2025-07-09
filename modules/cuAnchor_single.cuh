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
