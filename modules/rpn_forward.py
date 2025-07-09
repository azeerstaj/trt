import math
import torch
from torch import Tensor
from .torchvision_utils import (
    clip_boxes_to_image,
    remove_small_boxes,
    batched_nms
)
from collections import namedtuple

# def rpn_head_forward(features):
#     for feature in features:
#         t = self.conv(feature)
#         logits.append(self.cls_logits(t))
#         bbox_reg.append(self.bbox_pred(t))
#     return logits, bbox_reg

"""
def estimate_rpn_output_shape(features: dict, num_anchors_per_location: int, batch_size: int, post_nms_top_n: int):
    feature_shapes = [feat.shape for feat in features.values()]
    total_anchors = sum(num_anchors_per_location * H * W for (_, _, H, W) in feature_shapes)
    max_boxes_per_image = post_nms_top_n
    return (batch_size, max_boxes_per_image, 4)
"""

def estimate_rpn_proposal_counts(features, A: int = 3, pre_nms_top_n: int = 1000, post_nms_top_n: int = 1000) -> dict:
    feature_shapes = [feat.shape for feat in features]
    
    total_raw_anchors = sum(A * H * W for _, _, H, W in feature_shapes)
    approx_topk_total = sum(min(pre_nms_top_n, A * H * W) for _, _, H, W in feature_shapes)

    # Estimate drops
    after_score_thresh = int(approx_topk_total * 0.5)   # e.g., 50% filtered out
    after_nms = int(after_score_thresh * 0.5)           # e.g., 50% drop due to NMS

    expected = min(after_nms, post_nms_top_n)
    max_proposals = post_nms_top_n
    min_proposals = 0  # Worst case

    return {
        "min": min_proposals,
        "expected": expected,
        "max": max_proposals,
        "raw_anchors": total_raw_anchors,
        "topk_before_nms": approx_topk_total,
    }
 


def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int, W: int) -> Tensor:
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
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
    return targets


def encode_single(reference_boxes: Tensor, proposals: Tensor, weights=(1.0, 1.0, 1.0, 1.0)) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes

    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
    """
    dtype = reference_boxes.dtype
    device = reference_boxes.device
    weights = torch.as_tensor(weights, dtype=dtype, device=device)
    targets = encode_boxes(reference_boxes, proposals, weights)

    return targets

def encode(reference_boxes: list[Tensor], proposals: list[Tensor]) -> list[Tensor]:
    boxes_per_image = [len(b) for b in reference_boxes]
    reference_boxes = torch.cat(reference_boxes, dim=0)
    proposals = torch.cat(proposals, dim=0)
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
    boxes_per_image = [b.size(0) for b in boxes]
    concat_boxes = torch.cat(boxes, dim=0)
    box_sum = 0
    for val in boxes_per_image:
        box_sum += val
    if box_sum > 0:
        rel_codes = rel_codes.reshape(box_sum, -1)
    pred_boxes = decode_single(rel_codes, concat_boxes)
    if box_sum > 0:
        pred_boxes = pred_boxes.reshape(box_sum, -1, 4)
    return pred_boxes


def concat_box_prediction_layers(box_cls: list[Tensor], box_regression: list[Tensor]) -> tuple[Tensor, Tensor]:
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression

def _topk_min(input: Tensor, orig_kval: int, axis: int) -> int:
    if not torch.jit.is_tracing():
        return min(orig_kval, input.size(axis))
    axis_dim_val = torch._shape_as_tensor(input)[axis].unsqueeze(0)
    min_kval = torch.min(torch.cat((torch.tensor([orig_kval], dtype=axis_dim_val.dtype), axis_dim_val), 0))
    # return _fake_cast_onnx(min_kval)
    return min_kval

def _get_top_n_idx(objectness: Tensor, num_anchors_per_level: list[int], pre_nms_top_n=1000) -> Tensor:
    r = []
    offset = 0
    for ob in objectness.split(num_anchors_per_level, 1):
        num_anchors = ob.shape[1]
        pre_nms_top_n = _topk_min(ob, pre_nms_top_n, 1)
        _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
        r.append(top_n_idx + offset)
        offset += num_anchors
    return torch.cat(r, dim=1)

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

    image_range = torch.arange(num_images, device=device)
    batch_idx = image_range[:, None]

    objectness = objectness[batch_idx, top_n_idx]
    levels = levels[batch_idx, top_n_idx]
    proposals = proposals[batch_idx, top_n_idx]

    objectness_prob = torch.sigmoid(objectness)

    final_boxes = []
    final_scores = []
    for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
        boxes = clip_boxes_to_image(boxes, img_shape)

        # remove small boxes
        keep = remove_small_boxes(boxes, min_size)
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

        # remove low scoring boxes
        # use >= for Backwards compatibility
        keep = torch.where(scores >= score_thresh)[0]
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

        # non-maximum suppression, independently done per level
        keep = batched_nms(boxes, scores, lvl, nms_thresh)

        # keep only topk scoring predictions
        keep = keep[: post_nms_top_n]
        boxes, scores = boxes[keep], scores[keep]

        final_boxes.append(boxes)
        final_scores.append(scores)
    return final_boxes, final_scores


def rpn_forward(images, features, anchors, objectness, pred_bbox_deltas):
    # RPN uses all feature maps that are available
    # features = list(features.values())
    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    proposals = decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    boxes, scores = filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
    return boxes

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
        torch.randn(batch_size, 256, 25, 25),
        torch.randn(batch_size, 256, 13, 13),
    ]

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

    # Run RPN forward
    proposals = rpn_forward(images, features, anchors, objectness, pred_bbox_deltas)
    estimates = estimate_rpn_proposal_counts(features)   
    print(estimates)

    print("Total Proposals:", len(proposals))

    for i, (boxes_per_image) in enumerate(proposals):
        print(f"Image {i}: {len(boxes_per_image)} proposals")
        print(boxes_per_image[:5])  # show top 5 proposals
