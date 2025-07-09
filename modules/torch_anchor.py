import torch

def anchor_forward(image, feature_maps: list[torch.Tensor], SIZES, ASPECT_RATIOS) -> torch.Tensor:
    grid_sizes = [fm.shape[-2:] for fm in feature_maps]
    image_size = image.shape[-2:]
    dtype = feature_maps[0].dtype
    device = feature_maps[0].device

    # Compute strides
    strides = [
        [
            torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
            torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device),
        ]
        for g in grid_sizes
    ]

    # Generate zero-centered anchors
    cell_anchors = []
    for sizes, aspect_ratios in zip(SIZES, ASPECT_RATIOS):
        scales = torch.as_tensor(sizes, dtype=dtype, device=device)
        ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(ratios)
        w_ratios = 1.0 / h_ratios
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        cell_anchors.append(base_anchors.round())

    anchors_all = []
    for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
        gh, gw = size
        sh, sw = stride
        shifts_x = torch.arange(0, gw, dtype=torch.int32, device=device) * sw
        shifts_y = torch.arange(0, gh, dtype=torch.int32, device=device) * sh
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
        anchors = (shifts[:, None, :] + base_anchors[None, :, :]).reshape(-1, 4)
        anchors_all.append(anchors)

    # Return anchors for batch[0] only (ONNX doesn’t support loops over batch)
    return torch.cat(anchors_all, dim=0)

