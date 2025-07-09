import torch
from torch import Tensor

import torch
import torch.nn.functional as F


def nms_pytorch(dets, scores, iou_threshold):
    """
    Pure PyTorch implementation of Non-Maximum Suppression (NMS).
    
    Args:
        dets (torch.Tensor): Bounding boxes tensor of shape (N, 4) where each row is [x1, y1, x2, y2]
        scores (torch.Tensor): Confidence scores tensor of shape (N,)
        iou_threshold (float): IoU threshold for suppression
    
    Returns:
        torch.Tensor: Indices of kept boxes (shape: [num_kept])
    """
    # Input validation
    assert dets.dim() == 2, f"boxes should be a 2d tensor, got {dets.dim()}D"
    assert dets.size(1) == 4, f"boxes should have 4 elements in dimension 1, got {dets.size(1)}"
    assert scores.dim() == 1, f"scores should be a 1d tensor, got {scores.dim()}D"
    assert dets.size(0) == scores.size(0), f"boxes and scores should have same number of elements in dimension 0, got {dets.size(0)} and {scores.size(0)}"
    
    # Handle empty input
    if dets.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=dets.device)
    
    # Move to CPU if needed (like the original C++ implementation)
    device = dets.device
    if dets.is_cuda:
        dets = dets.cpu()
        scores = scores.cpu()
    
    # Handle quantized tensors by dequantizing
    if dets.is_quantized:
        dets = dets.dequantize()
    if scores.is_quantized:
        scores = scores.dequantize()
    
    # Ensure same dtype
    if dets.dtype != scores.dtype:
        scores = scores.to(dets.dtype)
    
    ndets = dets.size(0)
    
    # Extract coordinates
    x1 = dets[:, 0]
    y1 = dets[:, 1] 
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    
    # Sort by scores in descending order
    order = scores.argsort(descending=True)
    
    # Calculate areas
    areas = (x2 - x1) * (y2 - y1)
    
    # Initialize suppression mask and keep list
    suppressed = torch.zeros(ndets, dtype=torch.bool, device=dets.device)
    keep = []
    
    for i in range(ndets):
        idx = order[i]
        if suppressed[idx]:
            continue
            
        keep.append(idx)
        
        if i == ndets - 1:  # Last box
            break
            
        # Get coordinates of current box
        ix1 = x1[idx].item()
        iy1 = y1[idx].item()
        ix2 = x2[idx].item()
        iy2 = y2[idx].item()
        iarea = areas[idx].item()
        
        # Check remaining boxes
        remaining_indices = order[i + 1:]
        remaining_mask = ~suppressed[remaining_indices]
        
        if remaining_mask.any():
            remaining_boxes = remaining_indices[remaining_mask]
            
            # Vectorized IoU computation for remaining boxes
            xx1 = torch.maximum(torch.tensor(ix1, device=dets.device), x1[remaining_boxes])
            yy1 = torch.maximum(torch.tensor(iy1, device=dets.device), y1[remaining_boxes])
            xx2 = torch.minimum(torch.tensor(ix2, device=dets.device), x2[remaining_boxes])
            yy2 = torch.minimum(torch.tensor(iy2, device=dets.device), y2[remaining_boxes])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            ovr = inter / (iarea + areas[remaining_boxes] - inter)
            
            # Mark boxes with high overlap as suppressed
            suppress_mask = ovr > iou_threshold
            suppressed[remaining_boxes[suppress_mask]] = True
    
    result = torch.tensor(keep, dtype=torch.long, device=dets.device)
    
    # Move back to original device if needed
    if device != dets.device:
        result = result.to(device)
    
    return result


def nms(boxes, scores, iou_threshold):
    """
    Drop-in replacement for torchvision.ops.nms
    
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).
    NMS iteratively removes lower scoring boxes which have an
    IoU greater than ``iou_threshold`` with another (higher scoring)
    box.
    
    Args:
        boxes (Tensor[N, 4]): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold
    
    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    # Input validation
    assert boxes.dim() == 2, f"boxes should be a 2d tensor, got {boxes.dim()}D"
    assert boxes.size(1) == 4, f"boxes should have 4 elements in dimension 1, got {boxes.size(1)}"
    assert scores.dim() == 1, f"scores should be a 1d tensor, got {scores.dim()}D"
    assert boxes.size(0) == scores.size(0), f"boxes and scores should have same number of elements in dimension 0, got {boxes.size(0)} and {scores.size(0)}"
    
    # Handle empty input
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    
    # Handle quantized tensors
    if boxes.is_quantized:
        boxes = boxes.dequantize()
    if scores.is_quantized:
        scores = scores.dequantize()
    
    # Ensure same dtype
    if boxes.dtype != scores.dtype:
        scores = scores.to(boxes.dtype)
    
    # Extract coordinates
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    # Calculate areas
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by scores in descending order
    order = scores.argsort(descending=True)
    
    keep = []
    while order.numel() > 0:
        # Pick the box with highest score
        idx = order[0]
        keep.append(idx)
        
        if order.numel() == 1:
            break
            
        # Get remaining boxes
        rest = order[1:]
        
        # Calculate IoU between current box and all remaining boxes
        xx1 = torch.maximum(x1[idx], x1[rest])
        yy1 = torch.maximum(y1[idx], y1[rest])
        xx2 = torch.minimum(x2[idx], x2[rest])
        yy2 = torch.minimum(y2[idx], y2[rest])
        
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        
        ovr = inter / (areas[idx] + areas[rest] - inter)
        
        # Keep boxes with IoU less than threshold
        order = rest[ovr <= iou_threshold]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
#     return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def _batched_nms_vanilla(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    # Based on Detectron2 implementation, just manually call nms() on each class independently
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(idxs):
        curr_indices = torch.where(idxs == class_id)[0]
        curr_keep_indices = nms(boxes[curr_indices], scores[curr_indices], iou_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]

def clip_boxes_to_image(boxes: Tensor, size: tuple[int, int]) -> Tensor:
    """
    Clip boxes so that they lie inside an image of size ``size``.

    .. note::
        For clipping a :class:`~torchvision.tv_tensors.BoundingBoxes` object, consider using
        the transform :func:`~torchvision.transforms.v2.ClampBoundingBoxes` instead.

    Args:
        boxes (Tensor[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        size (Tuple[height, width]): size of the image

    Returns:
        Tensor[N, 4]: clipped boxes
    """
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(clip_boxes_to_image)

    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    # if torchvision._is_tracing():
    #     boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
    #     boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
    #     boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
    #     boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
    # else:
    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """
    Remove every box from ``boxes`` which contains at least one side length
    that is smaller than ``min_size``.

    .. note::
        For sanitizing a :class:`~torchvision.tv_tensors.BoundingBoxes` object, consider using
        the transform :func:`~torchvision.transforms.v2.SanitizeBoundingBoxes` instead.

    Args:
        boxes (Tensor[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        min_size (float): minimum size

    Returns:
        Tensor[K]: indices of the boxes that have both sides
        larger than ``min_size``
    """
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(remove_small_boxes)
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    keep = torch.where(keep)[0]
    return keep

def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    return _batched_nms_vanilla(boxes, scores, idxs, iou_threshold)
