import random
import math
import torch
from torch import Tensor
from .torchvision_utils import (
    clip_boxes_to_image,
    remove_small_boxes,
    batched_nms
)
from collections import namedtuple

torch.manual_seed(0)

def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int, W: int) -> Tensor:
    print(f"[permute_and_flatten] input: {layer.shape}")
    # print("")
    layer = layer.view(N, -1, C, H, W) # BUG
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    print(f"[permute_and_flatten] output: {layer.shape}")
    return layer

def encode_boxes(reference_boxes: Tensor, proposals: Tensor, weights: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes

    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
        weights (Tensor[4]): the weights for ``(x, y, w, h)``
    """
    print(f"[encode_boxes] reference_boxes: {reference_boxes.shape}, proposals: {proposals.shape}")


    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    print(f"[encode_boxes] output: {targets.shape}")
    return targets


def encode_single(reference_boxes: Tensor, proposals: Tensor, weights=(1.0, 1.0, 1.0, 1.0)) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes

    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
    """
    print(f"[encode_single] input: {reference_boxes.shape}, {proposals.shape}")
    dtype = reference_boxes.dtype
    device = reference_boxes.device
    weights = torch.as_tensor(weights, dtype=dtype, device=device)
    targets = encode_boxes(reference_boxes, proposals, weights)
    print(f"[encode_single] output: {targets.shape}")

    return targets

def encode(reference_boxes: list[Tensor], proposals: list[Tensor]) -> list[Tensor]:
    print(f"[encode] input: {len(reference_boxes)} images")
    boxes_per_image = [len(b) for b in reference_boxes]
    reference_boxes = torch.cat(reference_boxes, dim=0)
    proposals = torch.cat(proposals, dim=0)
    # print(f"[encode] cat reference: {proposals.shape}, cat proposals: {proposals_cat.shape}")
    targets = encode_single(reference_boxes, proposals)
    return targets.split(boxes_per_image, 0)

def decode_single(rel_codes: Tensor, boxes: Tensor, weights=(1.0, 1.0, 1.0, 1.0),
                   bbox_xform_clip = math.log(1000.0 / 16)) -> Tensor:
    """
    From a set of original boxes and encoded relative box offsets,
    get the decoded boxes.

    Args:
        rel_codes (Tensor): encoded boxes
        boxes (Tensor): reference boxes.
    """
    print(f"[decode_single] input: rel_codes={rel_codes.shape}, boxes={boxes.shape}")

    boxes = boxes.to(rel_codes.dtype)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = rel_codes[:, 0::4] / wx
    dy = rel_codes[:, 1::4] / wy
    dw = rel_codes[:, 2::4] / ww
    dh = rel_codes[:, 3::4] / wh

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=bbox_xform_clip)
    dh = torch.clamp(dh, max=bbox_xform_clip)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    # Distance from center to box's corner.
    c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
    c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w

    pred_boxes1 = pred_ctr_x - c_to_c_w
    pred_boxes2 = pred_ctr_y - c_to_c_h
    pred_boxes3 = pred_ctr_x + c_to_c_w
    pred_boxes4 = pred_ctr_y + c_to_c_h
    pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
    print(f"[decode_single] output: {pred_boxes.shape}")
    return pred_boxes


def decode(rel_codes: Tensor, boxes: list[Tensor]) -> Tensor:
    torch._assert(
        isinstance(boxes, (list, tuple)),
        f"This function expects boxes of type list or tuple, not {type(boxes)}",
    )
    torch._assert(
        isinstance(rel_codes, torch.Tensor),
        "This function expects rel_codes of type torch.Tensor.",
    )
    print(f"[decode] input: rel_codes={rel_codes.shape}, {len(boxes)} boxes")
    boxes_per_image = [b.size(0) for b in boxes]
    concat_boxes = torch.cat(boxes, dim=0)
    box_sum = 0
    for val in boxes_per_image:
        box_sum += val
    if box_sum > 0:
        rel_codes = rel_codes.reshape(box_sum, -1)
    print(f"[decode] concat_boxes: {concat_boxes.shape}, reshaped rel_codes: {rel_codes.shape}")
    pred_boxes = decode_single(rel_codes, concat_boxes)
    if box_sum > 0:
        pred_boxes = pred_boxes.reshape(box_sum, -1, 4)
    print(f"[decode] output: {pred_boxes.shape}")
    return pred_boxes


def concat_box_prediction_layers(box_cls: list[Tensor], box_regression: list[Tensor]) -> tuple[Tensor, Tensor]:
    print(f"[concat_box_prediction_layers] input: {len(box_cls)} levels")
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # print("box cls per lvl:", box_cls_per_level.shape)
        # print("box reg per lvl:", box_regression_per_level.shape)
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        # print("N:", N)
        # print("Ax4:", Ax4)
        # print("AxC:", AxC)
        # print("AxC // A:", AxC // A)
        # print("C:", C)
        # print("A:", A)
        # print("H:", H)
        # print("W:", W)
        # print("Box cls per lvl:", box_cls_per_level.shape)
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    # print(f"[concat_box_prediction_layers] output: box_cls={box_cls.shape}, box_regression={box_regression.shape}")
    return box_cls, box_regression

def _topk_min(input: Tensor, orig_kval: int, axis: int) -> int:
    if not torch.jit.is_tracing():
        return min(orig_kval, input.size(axis))
    axis_dim_val = torch._shape_as_tensor(input)[axis].unsqueeze(0)
    min_kval = torch.min(torch.cat((torch.tensor([orig_kval], dtype=axis_dim_val.dtype), axis_dim_val), 0))
    # return _fake_cast_onnx(min_kval)
    return min_kval

def _get_top_n_idx(objectness: Tensor, num_anchors_per_level: list[int], pre_nms_top_n=1000) -> Tensor:
    print(f"[_get_top_n_idx] input: {objectness.shape}")
    r = []
    offset = 0
    for ob in objectness.split(num_anchors_per_level, 1):
        num_anchors = ob.shape[1]
        pre_nms_top_n = _topk_min(ob, pre_nms_top_n, 1)
        _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
        r.append(top_n_idx + offset)
        offset += num_anchors
    result = torch.cat(r, dim=1)
    print(f"[_get_top_n_idx] output: {result.shape}")
    return result

def filter_proposals(
    proposals: Tensor,
    objectness: Tensor,
    image_shapes: list[tuple[int, int]],
    num_anchors_per_level: list[int],
    min_size=1e-3, # from class
    score_thresh = 0.0, # from class
    nms_thresh = 0.7,
    post_nms_top_n = 1000
) -> tuple[list[Tensor], list[Tensor]]:

    print(f"[filter_proposals] proposals: {proposals.shape}, objectness: {objectness.shape}")
    # print(f"[filter_proposals] proposals1:", proposals)
    num_images = proposals.shape[0]
    device = proposals.device
    # do not backprop through objectness
    objectness = objectness.detach()
    objectness = objectness.reshape(num_images, -1)

    levels = [
        torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
    ]
    levels = torch.cat(levels, 0)
    levels = levels.reshape(1, -1).expand_as(objectness)

    # select top_n boxes independently per level before applying nms
    top_n_idx = _get_top_n_idx(objectness, num_anchors_per_level)
    print(f"[filter_proposals] top_n_idx: {top_n_idx.shape}")

    image_range = torch.arange(num_images, device=device)
    batch_idx = image_range[:, None]

    objectness = objectness[batch_idx, top_n_idx]
    levels = levels[batch_idx, top_n_idx]
    proposals = proposals[batch_idx, top_n_idx]

    objectness_prob = torch.sigmoid(objectness)

    final_boxes = []
    final_scores = []
    for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
        print(f"[filter_proposals] before NMS: boxes={boxes.shape}, scores={scores.shape}, img_shape={img_shape}")#, level={lvl}")

        boxes = clip_boxes_to_image(boxes, img_shape)
        print("[filter_proposals] boxes2:", boxes.shape)
        # print(boxes)

        # remove small boxes
        keep = remove_small_boxes(boxes, min_size)
        print("[filter_proposals] boxes[:5]:", boxes[0][:5])
        print("[filter_proposals] len(keep1):", len(keep), ",min_size:", min_size)
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

        # remove low scoring boxes
        # use >= for Backwards compatibility
        keep = torch.where(scores >= score_thresh)[0]
        print("[filter_proposals] scores[:5]", scores[:5])
        print("[filter_proposals] len(keep2):", len(keep), ",score_thresh:", score_thresh)
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

        # non-maximum suppression, independently done per level
        keep = batched_nms(boxes, scores, lvl, nms_thresh)
        print("[filter_proposals] len(keep3):", len(keep))

        # keep only topk scoring predictions
        keep = keep[: post_nms_top_n]
        print("[filter_proposals] len(keep4):", len(keep))
        boxes, scores = boxes[keep], scores[keep]

        final_boxes.append(boxes)
        final_scores.append(scores)
        print(f"[filter_proposals] after NMS: boxes={boxes.shape}")
    return final_boxes, final_scores

def rpn_forward(images, anchors, objectness, pred_bbox_deltas):
    if objectness[0].ndim == 3:
        objectness = [o.unsqueeze(0) for o in objectness]
    if pred_bbox_deltas[0].ndim == 3:
        pred_bbox_deltas = [p.unsqueeze(0) for p in pred_bbox_deltas]
    print(f"[rpn_forward] len(images): {len(images.image_sizes)}")
    print(f"[rpn_forward] len(objectness): {len(objectness)}")
    print(f"[rpn_forward] len(pred_bbox_deltas): {len(pred_bbox_deltas)}")
    print(f"[rpn_forward] objectness[0].shape: {objectness[0].shape}")
    print(f"[rpn_forward] pred_bbox_deltas[0].shape: {pred_bbox_deltas[0].shape}")
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    print(f"[rpn_forward] num_anchors_per_level_shape_tensors: {num_anchors_per_level_shape_tensors}")
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    print(f"[rpn_forward] objectness: {objectness.shape}, pred_bbox_deltas: {pred_bbox_deltas.shape}")
    print(f"[rpn_forward] objectness: {objectness.dtype}, pred_bbox_deltas: {pred_bbox_deltas.dtype}")

    proposals = decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(len(anchors), -1, 4)
    print(f"[rpn_forward] proposals reshaped: {proposals.shape}")
    print(f"[rpn_forward] proposals reshaped: {proposals.dtype}")

    boxes, scores = filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
    print(f"[rpn_forward] final boxes: {[b.shape for b in boxes]}")
    return [b.cpu().numpy() for b in boxes] # sussy 

ImageList = namedtuple("ImageList", ["tensors", "image_sizes"])
if __name__ == "__main__":
    # Simulated input
    batch_size = 1
    image_size = (800, 800)
    image_tensor = torch.randn(batch_size, 3, *image_size)
    images = ImageList(tensors=image_tensor, image_sizes=[image_size])

    # Simulate features from 3 FPN levels
    features = [
        torch.randn(batch_size, 256, 50, 50),
        torch.randn(batch_size, 512, 25, 25),
        torch.randn(batch_size, 1024, 10, 10),
        torch.randn(batch_size, 1024, 10, 10),
    ]

    # features = [
    #     torch.randn(256, 50, 50),
    #     torch.randn(512, 25, 25),
    #     torch.randn(1024, 10, 10),
    #     torch.randn(1024, 10, 10),
    # ]
    # Assume 3 anchors per spatial location
    num_anchors = 3
    objectness = []
    pred_bbox_deltas = []

    for feat in features:
        N, C, H, W = feat.shape
        objectness.append(torch.randn(N, num_anchors, H, W))
        pred_bbox_deltas.append(torch.randn(N, num_anchors * 4, H, W))

    # Simulate anchors for each image
    anchors = []
    for _ in range(batch_size):
        all_anchors = []
        for feat in features:
            _, _, H, W = feat.shape
            A = num_anchors
            total = H * W * A
            anchors_per_feat = torch.rand(total, 4) * 800  # random box coords
            all_anchors.append(anchors_per_feat)
        anchors.append(torch.cat(all_anchors, dim=0))

    print("len(anchors):", len(anchors))
    print("anchors[0].shape:", anchors[0].shape)
    
    print("len(objectness):", len(objectness))
    print("objectness[0].shape:", objectness[0].shape)
    
    print("len(pred_bbox_deltas)", len(pred_bbox_deltas))
    print("pred_bbox_deltas[0].shape", pred_bbox_deltas[0].shape)

    print("\nANCHORS:\n", anchors[0][0][:5])
    print("\nOBJECTNESS:\n", objectness[0][0][0][0][:5])
    print("\nPRED_BBOX_DELTAS:\n", pred_bbox_deltas[0][0][0][0][:5])

    # Run RPN forward
    proposals = rpn_forward(images, anchors, objectness, pred_bbox_deltas)

    print("Total Proposals:", len(proposals))