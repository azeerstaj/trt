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
    int feat_h, int feat_w,       
    int stride_h, int stride_w,
    int anchor_counts
);
'''

# Load the CUDA extension
anchor_extension = load_inline(
    name='anchor_generation_single_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['generate_anchors_single_cuda'],# 'generate_anchors_single_level_cuda'],
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"]
)

def generate_anchors_single_pytorch(img_h, img_w, feat_h, feat_w, sizes, aspect_ratios):
    """
    PyTorch wrapper that mimics your original NumPy function
    """
    # Prepare base anchors for all levels
    base_anchors_list = []
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
        total_base_anchors += num_anchors
        base_anchors_list.extend(level_base_anchors)
    
    # Convert to tensors
    base_anchors = torch.tensor(base_anchors_list, dtype=torch.float32, device='cuda').view(-1, 4)
    base_anchors = torch.round(base_anchors)  # Match NumPy behavior
    stride_h = img_h // feat_h
    stride_w = img_w // feat_w
    
    # Call CUDA kernel
    result = anchor_extension.generate_anchors_single_cuda(
        base_anchors,
        feat_h, feat_w,
        stride_h, stride_w,
        total_base_anchors
    )
    
    return result

# Example usage
if __name__ == "__main__":
    # Test parameters
    image_height, image_width = 800, 800
    feature_height, feature_width = 20, 20

    sizes = [[32], [64], [128]]
    aspect_ratios = [[0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]]
    
    # Generate anchors using CUDA
    anchors_cuda = generate_anchors_single_pytorch(
        image_height, image_width,
        feature_height, feature_width,
        sizes, aspect_ratios
    )
    
    print(f"Generated {anchors_cuda.shape[0]} anchors")
    print(f"First 5 anchors:\n{anchors_cuda[:5]}")
