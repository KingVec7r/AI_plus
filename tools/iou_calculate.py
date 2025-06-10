import torch
from torchvision.ops import box_iou
boxes_a = torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]])
boxes_b = torch.tensor([[100, 125, 200, 225], [350, 350, 450, 450]])
iou = box_iou(boxes_a, boxes_b)  # 输出形状为(2, 2)
print(iou)