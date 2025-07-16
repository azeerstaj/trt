import torch
from torch import Tensor
import torch.nn.functional as F

def rpn_head_forward(x: list[Tensor], weights):
    logits, bbox_reg= [], []
    for feature in x:

        print(f"Processing feature map of shape: {feature.shape}")
        t = F.conv2d(input=feature, weight=weights['rpn.head.conv.0.0.weight'],
                bias=weights['rpn.head.conv.0.0.bias'], padding=1)

        cls_logits = F.conv2d(input=t, weight=weights['rpn.head.cls_logits.weight'],
                               bias=weights['rpn.head.cls_logits.bias'])

        bbox_pred = F.conv2d(input=t, weight=weights['rpn.head.bbox_pred.weight'],
                             bias=weights['rpn.head.bbox_pred.bias'])

        logits.append(cls_logits.squeeze().cpu())
        bbox_reg.append(bbox_pred.squeeze().cpu())

    return logits, bbox_reg

if __name__ == "__main__":
    weights = torch.load("../weights/fasterrcnn1.pt", weights_only=True, map_location='cuda')

    image_shape = [1, 3, 800, 800]
    f1_shape = [1, 256, 200, 200]
    f2_shape = [1, 256, 100, 100]
    f3_shape = [1, 256, 50, 50]
    f4_shape = [1, 256, 25, 25]
    f5_shape = [1, 256, 13, 13]

    fmaps = [f1_shape, f2_shape, f3_shape, f4_shape, f5_shape]
    fmaps = [torch.randn(shape, dtype=torch.float32, device='cuda') for shape in fmaps]

    logits, bbox_reg = rpn_head_forward(fmaps, weights)
    print("logits:", logits[0].shape)
    print("bbox:", bbox_reg[0].shape)
    # print("sa")