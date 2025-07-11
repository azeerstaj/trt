import torch
from torch import Tensor
import torch.nn.functional as F

def rpn_head_forward(x: list[Tensor], weights):
    logits, bbox_reg= [], []
    for feature in x:

        t = F.conv2d(input=feature, weight=weights['rpn.head.conv.0.0.weight'],
                bias=weights['rpn.head.conv.0.0.bias'], padding=1)

        cls_logits = F.conv2d(input=t, weight=weights['rpn.head.cls_logits.weight'],
                               bias=weights['rpn.head.cls_logits.bias'])

        bbox_pred = F.conv2d(input=t, weight=weights['rpn.head.bbox_pred.weight'],
                             bias=weights['rpn.head.bbox_pred.bias'])

        logits.append(cls_logits.cpu())
        bbox_reg.append(bbox_pred.cpu())

    return logits, bbox_reg

if __name__ == "__main__":
    weights = torch.load("../weights/fasterrcnn1.pt", weights_only=True, map_location='cuda')
    x = torch.randn(1, 256, 50, 50).cuda()
    out1, out2 = rpn_head_forward(x, weights)
    print(out1[0].shape)
    # print("sa")