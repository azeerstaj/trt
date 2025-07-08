import torch
from modules.cu_anchor_single import generate_anchors_single_pytorch
from modules.cu_anchor import generate_anchors_pytorch
from modules.numpy_anchor import anchor_forward_numpy


if __name__ == "__main__":
    image_shape = (1, 3, 800, 600)

    dummy_f4 = torch.randn(1, 256, 20, 20)
    # feature_maps = [dummy_f4]#dummy_f2, dummy_f3]

    SIZES = ((32,),)
    ASPECT_RATIOS = ((0.5,),)

    # Prepare input for numpy version
    image_height, image_width = image_shape[-2], image_shape[-1]
    feature_map_shapes = [dummy_f4.shape[-2:]]  # [(100, 100), (50, 50), (25, 25)]

    # Call the numpy anchor generator
    anchors_np = anchor_forward_numpy(
        image_height=image_height,
        image_width=image_width,
        feature_map_shapes=[dummy_f4.shape[-2:]],
        sizes=SIZES,
        aspect_ratios=ASPECT_RATIOS
    )

    anchors_1 = generate_anchors_single_pytorch(image_height,
                                                image_width,
                                                dummy_f4.shape[-2],
                                                dummy_f4.shape[-1],
                                                SIZES,
                                                ASPECT_RATIOS)


    anchors_2 = generate_anchors_pytorch(image_height,
                                        image_width,
                                        feature_map_shapes,
                                        SIZES,
                                        ASPECT_RATIOS)

    anchors_ref = torch.tensor(anchors_np, dtype=anchors_1.dtype, device=anchors_1.device)

    print(f"First 5 anchors:\n{anchors_1[:5]}")
    print(f"First 5 anchors:\n{anchors_2[:5]}")
    print(f"First 5 anchors:\n{anchors_ref[:5]}")

    print("Single and NumPy is the same !" if torch.allclose(anchors_ref, anchors_1) else "Single and NumPy is not the Same ...")
    print("Multi and NumPy is the same !" if torch.allclose(anchors_ref, anchors_2) else "Single and NumPy is not the Same ...")