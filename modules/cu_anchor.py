import os
import torch
from torch.utils.cpp_extension import load_inline
import numpy as np

# Read CUDA kernel from file
def read_cuda_kernel(filename):
    with open(filename, 'r') as f:
        return f.read()

# Load the CUDA kernel source
base_path = "/home/baykar/git/anchor-gen/modules"
cuda_source = read_cuda_kernel(os.path.join(base_path,'cu_anchor.cpp'))

# C++ function declarations
cpp_source = '''
torch::Tensor generate_anchors_cuda(
    torch::Tensor base_anchors,
    torch::Tensor feature_map_info,
    torch::Tensor anchor_counts,
    torch::Tensor level_offsets,
    torch::Tensor output_offsets,
    int total_output_anchors
);
'''

# Load the CUDA extension
anchor_extension = load_inline(
    name='anchor_generation_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['generate_anchors_cuda'],
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"]
)

def generate_anchors_pytorch(image_height, image_width, feature_map_shapes, sizes, aspect_ratios):
    """
    PyTorch wrapper that mimics your original NumPy function
    """
    # Prepare base anchors for all levels
    base_anchors_list = []
    anchor_counts = []
    level_offsets = []
    
    total_base_anchors = 0
    for scale_set, ratio_set in zip(sizes, aspect_ratios):
        level_base_anchors = []
        for scale in scale_set:
            for ratio in ratio_set:
                h = scale * np.sqrt(ratio)
                w = scale / np.sqrt(ratio)
                x1 = -w / 2
                y1 = -h / 2
                x2 = w / 2
                y2 = h / 2
                level_base_anchors.extend([x1, y1, x2, y2])
        
        num_anchors = len(level_base_anchors) // 4
        level_offsets.append(total_base_anchors)
        anchor_counts.append(num_anchors)
        total_base_anchors += num_anchors
        base_anchors_list.extend(level_base_anchors)
    
    # Convert to tensors
    base_anchors = torch.tensor(base_anchors_list, dtype=torch.float32, device='cuda').view(-1, 4)
    base_anchors = torch.round(base_anchors)  # Match NumPy behavior
    
    # Prepare feature map info
    feature_map_info = []
    output_offsets = []
    total_output_anchors = 0
    
    for i, (feat_h, feat_w) in enumerate(feature_map_shapes):
        stride_h = image_height // feat_h
        stride_w = image_width // feat_w
        feature_map_info.extend([feat_h, feat_w, stride_h, stride_w])
        
        level_output_anchors = feat_h * feat_w * anchor_counts[i]
        total_output_anchors += level_output_anchors
        output_offsets.append(total_output_anchors)
    
    # Convert to tensors
    feature_map_info = torch.tensor(feature_map_info, dtype=torch.int32, device='cuda').view(-1, 4)
    anchor_counts = torch.tensor(anchor_counts, dtype=torch.int32, device='cuda')
    level_offsets = torch.tensor(level_offsets, dtype=torch.int32, device='cuda')
    output_offsets = torch.tensor(output_offsets, dtype=torch.int32, device='cuda')
    
    # Call CUDA kernel
    result = anchor_extension.generate_anchors_cuda(
        base_anchors,
        feature_map_info,
        anchor_counts,
        level_offsets,
        output_offsets,
        total_output_anchors
    )
    
    return result

# Example usage
if __name__ == "__main__":
    # Test parameters
    image_height, image_width = 800, 600
    feature_map_shapes = [(20, 20)]#, (50, 38), (25, 19)]
    sizes = [[32], [64], [128]]
    aspect_ratios = [[0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]]
    
    # Generate anchors using CUDA
    anchors_cuda = generate_anchors_pytorch(
        image_height, image_width, feature_map_shapes, sizes, aspect_ratios
    )
    
    print(f"Generated {anchors_cuda.shape[0]} anchors")
    print(f"First 5 anchors:\n{anchors_cuda[:5]}")