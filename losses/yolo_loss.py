from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.calculate_iou import bbox_wh_iou, bbox_iou


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    """依据网络的预测值和真实值计算与各个anchor所匹配的ground truth

    Args:
        pred_boxes: 预测的边界框, [batch_size, number_anchors, grid_x, grid_y, 4]
        pred_cls: 预测的类别概率, [batch_size, number_anchors, grid_x, grid_y, number_classes]
        target: 真实边界框, [number_all_bboxes, 6], 将每个batch的所有bboxes在第0维进行拼接
        anchors: anchor的宽和高， [9, 2]
        ignore_thres: float, 当anchor与target的iou高于该阈值时，才认为该anchor存在目标
    Return:
        iou_scores: 表示每一个anchor和与其匹配的target的iou
        class_mask: 表示每一个anchor的类别预测是否正确(bool)
        object_mask： 每一个元素（bool）表示对应的anchor是否有target与其对应 
        noobject_mask： 与object_mask的含义相反
        target_x: 真实边界框的x坐标，与每一个anchor对应
        target_y: 真实边界框的y坐标，与每一个anchor对应
        target_w: 真实边界框的宽，与每一个anchor对应
        target_h: 真实边界框的高，与每一个anchor对应
        target_cls: 真实边界框对应的类别，与每一个anchor对应 
        target_confidence: 是否存在目标，与每一个anchor对应
    """
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    # 尺寸信息
    batch_size = pred_boxes.size(0)
    number_anchor = pred_boxes.size(1)
    number_classes = pred_cls.size(-1)
    number_grid = pred_boxes.size(2)

    # output tensors
    # 掩膜
    object_mask = ByteTensor(batch_size, number_anchor, number_grid, number_grid).fill_(0)
    noobject_mask = ByteTensor(batch_size, number_anchor, number_grid, number_grid).fill_(1)
    class_mask = FloatTensor(batch_size, number_anchor, number_grid, number_grid).fill_(0)
    iou_scores = FloatTensor(batch_size, number_anchor, number_grid, number_grid).fill_(0)
    # 真实target
    target_x = FloatTensor(batch_size, number_anchor, number_grid, number_grid).fill_(0)
    target_y = FloatTensor(batch_size, number_anchor, number_grid, number_grid).fill_(0)
    target_w = FloatTensor(batch_size, number_anchor, number_grid, number_grid).fill_(0)
    target_h = FloatTensor(batch_size, number_anchor, number_grid, number_grid).fill_(0)
    target_cls = FloatTensor(batch_size, number_anchor, number_grid, number_classes).fill_(0)

    # 将真实标定target转换为相对于box的位置
    target_bboxes = target[:, 2:6] * number_grid
    ground_xy = target_bboxes[:, :2]
    ground_wh = target_bboxes[:, 2:]
    # 计算
    ious = torch.stack([bbox_wh_iou(anchor, ground_wh) for anchor in anchors])
    best_ious, best_n = ious.max(0)

    sample_index, target_labels = target[:, :2].long().t()
    ground_x, ground_y = ground_xy.t()
    ground_w, ground_h = ground_wh.t()
    ground_i, ground_j = ground_xy.long().t()
    # 设置掩膜
    object_mask[sample_index, best_n, ground_j, ground_i] = 1
    noobject_mask[sample_index, best_n, ground_j, ground_i] = 0

    for index, anchor_ious in enumerate(ious.t()):
        noobject_mask[sample_index[index], anchor_ious > ignore_thres, ground_j[index], ground_i[index]] = 0
    
    # 对将真实标定值进行转换，x, y转换为偏移值, w h依据公式转换到log空间
    target_x[sample_index, best_n, ground_j, ground_i] = ground_x - ground_x.floor()
    target_y[sample_index, best_n, ground_j, ground_i] = ground_y - ground_y.floor()
    target_w[sample_index, best_n, ground_j, ground_i] = torch.log(ground_w / anchors[best_n][:, 0] + 1e-16)
    target_h[sample_index, best_n, ground_j, ground_i] = torch.log(ground_h / anchors[best_n][:, 1] + 1e-16)
    # 将类别转换为one-hot编码
    target_cls[sample_index, best_n, ground_j, ground_i, target_labels] = 1
    # 计算类别正确度和最佳anchor的iou
    class_mask[sample_index, best_n, ground_j, ground_i] = (pred_cls[sample_index, best_n, ground_j, ground_i].argmax(-1) == target_labels).float()
    iou_scores[sample_index, best_n, ground_j, ground_i] = bbox_iou(pred_boxes[sample_index, best_n, ground_j, ground_i], target_bboxes, x1y1x2y2=False)

    target_confidence = object_mask.float()
    return iou_scores, class_mask, object_mask, noobject_mask, target_x, target_y, target_w, target_h, target_cls, target_confidence