import torch
# from torchvision.models.detection.anchor_utils import AnchorGenerator
# from torchvision.models.detection.image_list import ImageList

# Anchor generator parameters
SIZES = [(32,), (64,), (128,)]
ASPECT_RATIOS = [(0.5, 1.0, 2.0)] * len(SIZES)

import numpy as np

def anchor_forward_numpy(image_height, image_width, feature_map_shapes, sizes, aspect_ratios):
    anchors_all = []

    for (feat_h, feat_w), scale_set, ratio_set in zip(feature_map_shapes, sizes, aspect_ratios):
        # Compute stride (integer division)
        stride_h = image_height // feat_h
        stride_w = image_width // feat_w

        # === Base anchors (zero-centered) ===
        base_anchors = []
        for scale in scale_set:
            for ratio in ratio_set:
                h = scale * np.sqrt(ratio)
                w = scale / np.sqrt(ratio)
                x1 = -w / 2
                y1 = -h / 2
                x2 = w / 2
                y2 = h / 2
                base_anchors.append([x1, y1, x2, y2])
        
        # Match PyTorch behavior: round and keep float32
        base_anchors = np.round(np.array(base_anchors, dtype=np.float32))  # shape: (A, 4)

        # === Generate grid shifts ===
        shift_x = np.arange(0, feat_w, dtype=np.float32) * stride_w  # shape: (W,)
        shift_y = np.arange(0, feat_h, dtype=np.float32) * stride_h  # shape: (H,)
        # print("Shift X:", shift_x)
        # print("Shift Y:", shift_y)

        shift_y, shift_x = np.meshgrid(shift_y, shift_x, indexing="ij")  # shape: (H, W)

        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        shifts = np.stack([shift_x, shift_y, shift_x, shift_y], axis=1)  # shape: (H*W, 4)
        # print("Shifts:", shifts[:10])

        # === Combine shifts and base anchors ===
        anchors = shifts[:, None, :] + base_anchors[None, :, :]  # (H*W, A, 4)
        anchors = anchors.reshape(-1, 4)  # shape: (H*W * A, 4)
        # print("Anchors:", anchors[:10])
        # exit(0)

        anchors_all.append(anchors)

    return np.vstack(anchors_all).astype(np.float32)  # Final shape: (total_anchors, 4)


# === MAIN ===
if __name__ == "__main__":

    image_shape = (1, 3, 800, 800)
    dummy_images = torch.randn(image_shape)

    dummy_f2 = torch.randn(1, 256, 50, 50)
    dummy_f3 = torch.randn(1, 256, 25, 25)
    dummy_f4 = torch.randn(1, 256, 10, 10)
    feature_maps = [dummy_f4, dummy_f2, dummy_f3]

    # Prepare input for numpy version
    image_height, image_width = dummy_images.tensors.shape[-2:]
    feature_map_shapes = [tuple(fm.shape[-2:]) for fm in feature_maps]  

    # Call the numpy anchor generator
    anchors_2_np = anchor_forward_numpy(
        image_height=image_height,
        image_width=image_width,
        feature_map_shapes=feature_map_shapes,
        sizes=SIZES,
        aspect_ratios=ASPECT_RATIOS
    )

    """
    # Print a few anchors
    print("Ref:")
    print(anchors_ref[:5], "\n")

    print("Raw Anchors:")
    print(anchors_1[:5], "\n")

    print("Numpy Anchors:")
    print(anchors_2[:5], "\n")
    """
