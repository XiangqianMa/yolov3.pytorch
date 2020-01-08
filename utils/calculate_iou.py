import torch


def bbox_wh_iou(wh1, wh2):
    """给定两个box的宽和高，计算两者的IoU

    Args:
        wh1: [w, h]
        wh2: [w, h]
    Return:
        iou
    """
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """计算box1与box2之间的iou

    Args:
        box1: [number_boxes, 4], 4中的前两个元素为坐标，后两个元素为宽、高
        box2: [number_boxes, 4], 同上
        x1y1x2y2: bool, 标记box1和box2中的box的表示格式，左上坐标还是中心坐标
    """
    if not x1y1x2y2:
        # 将中心坐标转换为左上右下
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2   
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 计算相交区域的左上坐标和右下坐标，左上取x,y最大，右下取x,y最小
    inter_rect_x1 = torch.max(b1_x1, b2_x1)             
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # 相交区域的面积
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # 并的面积
    union_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1) + (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (union_area - inter_area + 1e-16)
    return iou