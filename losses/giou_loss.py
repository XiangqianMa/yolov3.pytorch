import torch
from torch import nn


class GIoULoss(nn.Module):
    """
    generalize interaction over union loss
    """
    def __init__(self, reduction='mean'):
        super(GIoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pr_bboxes, gt_bboxes):
        """

        Args:
            pr_bboxes: tensor (-1, 4)
            gt_bboxes: tensor (-1, 4)

        Returns:
            loss
        """
        gt_area = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        pr_area = (pr_bboxes[:, 2] - pr_bboxes[:, 0]) * (pr_bboxes[:, 3] - pr_bboxes[:, 1])

        # iou
        lt = torch.max(gt_bboxes[:, :2], pr_bboxes[:, :2])
        rb = torch.min(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
        TO_REMOVE = 1
        wh = (rb - lt + TO_REMOVE).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        union = gt_area + pr_area - inter
        iou = inter / union
        # enclosure
        lt = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
        rb = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
        wh = (rb - lt + TO_REMOVE).clamp(min=0)
        enclosure = wh[:, 0] * wh[:, 1]

        giou = iou - (enclosure - union) / enclosure
        loss = 1. - giou
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass

        return loss


def bbox_transfer(center_x, center_y, w, h):
    """
    (x, y, w, h) -> (x1, y1, x2, y2)
    Args:
        center_x: bbox中心坐标x
        center_y: bbox中心坐标y
        w: bbox的宽
        h: bbox的高

    Returns:
        bbox_converted: (batch_size, 4), x1y1x2y2
    """
    bbox_converted = torch.zeros(center_x.size[0], 4).type_as(center_x)
    bbox_converted[:, 0] = center_x - w / 2.0
    bbox_converted[:, 1] = center_y - h / 2.0
    bbox_converted[:, 2] = center_x + w / 2.0
    bbox_converted[:, 3] = center_y + h / 2.0

    return bbox_converted
