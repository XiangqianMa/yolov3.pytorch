from __future__ import division
import torch
import torch.nn as nn

from losses.focal_loss import FocalLoss
from losses.giou_loss import GIoULoss, bbox_transfer
from utils.calculate_iou import bbox_wh_iou, bbox_wh_giou, bbox_iou
from utils.util import to_cpu


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres, bbox_loss='raw', iou_type='iou'):
    """依据网络的预测值和真实值计算与各个anchor所匹配的ground truth

    Args:
        pred_boxes: 预测的边界框, [batch_size, number_anchors, grid_x, grid_y, 4]
        pred_cls: 预测的类别概率, [batch_size, number_anchors, grid_x, grid_y, number_classes]
        target: 真实边界框, [number_all_bboxes, 6], 将每个batch的所有bboxes在第0维进行拼接
        anchors: anchor的宽和高， [9, 2]
        ignore_thres: float, 除了与target的IoU最高的anchor之外，在剩余的anchor中，当anchor与target的iou高于该阈值时，不将其纳入置信
                        度损失的计算中。
        bbox_loss: 边界框损失类型，raw、GIoU
    Return:
        object_mask： 每一个元素（bool）表示对应的anchor是否有target与其对应 
        noobject_mask： 与object_mask的含义相反
        target_x: bbox_loss为raw时，为相对于cell左上角的偏差，bbox_loss为GIoU时，为相对于当前特征图左上角的偏差
        target_y: bbox_loss为raw时，为相对于cell左上角的偏差，bbox_loss为GIoU时，为相对于当前特征图左上角的偏差
        target_w: bbox_loss为raw时，为转换到log空间，bbox_loss为GIoU时，为相对于当前特征图的像素个数
        target_h: bbox_loss为raw时，为转换到log空间，bbox_loss为GIoU时，为相对于当前特征图的像素个数
        target_cls: 真实边界框对应的类别，与每一个anchor对应 
        target_confidence: 是否存在目标，与每一个anchor对应
        raw_target_w: 真实边界框的宽(比例形式)，与每一个anchor对应
        raw_target_h: 真实边界框的高(比例形式)，与每一个anchor对应
    """
    BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    # 尺寸信息
    batch_size = pred_boxes.size(0)
    number_anchor = pred_boxes.size(1)
    number_classes = pred_cls.size(-1)
    number_grid = pred_boxes.size(2)

    # output tensors
    # 掩膜，每一个预测框都对应一个 0/1
    object_mask = BoolTensor(batch_size, number_anchor, number_grid, number_grid).fill_(0)
    noobject_mask = BoolTensor(batch_size, number_anchor, number_grid, number_grid).fill_(1)
    class_mask = FloatTensor(batch_size, number_anchor, number_grid, number_grid).fill_(0)
    iou_scores = FloatTensor(batch_size, number_anchor, number_grid, number_grid).fill_(0)
    # 真实target（转换为相对于cell的偏移量）
    target_x = FloatTensor(batch_size, number_anchor, number_grid, number_grid).fill_(0)
    target_y = FloatTensor(batch_size, number_anchor, number_grid, number_grid).fill_(0)
    target_w = FloatTensor(batch_size, number_anchor, number_grid, number_grid).fill_(0)
    target_h = FloatTensor(batch_size, number_anchor, number_grid, number_grid).fill_(0)
    target_cls = FloatTensor(batch_size, number_anchor, number_grid, number_grid, number_classes).fill_(0)
    # 原始的比例格式的target的w、h
    raw_target_w = FloatTensor(batch_size, number_anchor, number_grid, number_grid).fill_(0)
    raw_target_h = FloatTensor(batch_size, number_anchor, number_grid, number_grid).fill_(0)

    # target：ratio -> 在当前特征图中的位置
    target_bboxes = target[:, 2:6] * number_grid
    ground_xy = target_bboxes[:, :2]
    ground_wh = target_bboxes[:, 2:]
    raw_ground_wh = target[:, 4:6]
    # 计算当前的anchor与ground_wh的iou（将anchor和ground_wh在左上角对齐）
    if iou_type == 'iou':
        ious = torch.stack([bbox_wh_iou(anchor, ground_wh) for anchor in anchors])
    elif iou_type == 'giou':
        ious = torch.stack([bbox_wh_giou(anchor, ground_wh) for anchor in anchors])
    # 与每个ground_truth的IoU最大的anchor
    best_ious, best_n = ious.max(0)

    sample_index, target_labels = target[:, :2].long().t()
    ground_x, ground_y = ground_xy.t()
    ground_w, ground_h = ground_wh.t()
    raw_ground_w, raw_ground_h = raw_ground_wh.t()
    ground_i, ground_j = ground_xy.long().t()
    # 设置掩膜，每个ground_truth由其中心所在的cell中的三个anchor中IoU最大的那个负责预测
    object_mask[sample_index, best_n, ground_j, ground_i] = 1
    noobject_mask[sample_index, best_n, ground_j, ground_i] = 0

    for index, anchor_ious in enumerate(ious.t()):
        noobject_mask[sample_index[index], anchor_ious > ignore_thres, ground_j[index], ground_i[index]] = 0

    if bbox_loss == 'raw':
        # 对真实标定值进行转换，x, y转换为相对于cell的偏移值, w h依据公式转换到log空间
        target_x[sample_index, best_n, ground_j, ground_i] = ground_x - ground_x.floor()
        target_y[sample_index, best_n, ground_j, ground_i] = ground_y - ground_y.floor()
        target_w[sample_index, best_n, ground_j, ground_i] = torch.log(ground_w / anchors[best_n][:, 0] + 1e-16)
        target_h[sample_index, best_n, ground_j, ground_i] = torch.log(ground_h / anchors[best_n][:, 1] + 1e-16)
    elif bbox_loss == 'GIoU':
        # 坐标为相对于特征图左上角的偏移量，宽高为占当前特征图的像素个数
        target_x[sample_index, best_n, ground_j, ground_i] = ground_x
        target_y[sample_index, best_n, ground_j, ground_i] = ground_y
        target_w[sample_index, best_n, ground_j, ground_i] = ground_w
        target_h[sample_index, best_n, ground_j, ground_i] = ground_h

    raw_target_w[sample_index, best_n, ground_j, ground_i] = raw_ground_w
    raw_target_h[sample_index, best_n, ground_j, ground_i] = raw_ground_h

    # 将类别转换为one-hot编码
    target_cls[sample_index, best_n, ground_j, ground_i, target_labels] = 1
    target_confidence = object_mask.float()

    return object_mask, noobject_mask, target_x, target_y, target_w, target_h, target_cls, target_confidence, \
           raw_target_w, raw_target_h


class YOLOLoss(nn.Module):
    """计算损失，并打印统计参数
    """
    def __init__(self, ignore_thres=0.7, object_scale=1, noobject_scale=100, bbox_loss='raw', iou_type='iou'):
        """
        
        Args:
            ignore_thres: 当anchor与target的iou高于该阈值时，才认为该anchor存在目标
            object_scale: 有目标的预测框的损失权重

        """
        super(YOLOLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.focal_loss = FocalLoss(gamma=2, alpha=1.0, reduction='none')
        self.giou_loss = GIoULoss(reduction='none')

        self.object_scale = object_scale
        self.noobject_scale = noobject_scale
        self.ignore_thres = ignore_thres
        self.metric = []

        self.bbox_loss = bbox_loss
        self.iou_type = iou_type
    
    def forward(self, predicts, targets):
        loss = 0
        # 对Yolo Layer的输出依次计算损失
        for predict in predicts:
            predict_bboxes = predict[0]
            classes_probablity = predict[1]
            anchor_vector = predict[2] 
            center_x = predict[3] 
            center_y = predict[4]
            width = predict[5]
            height = predict[6] 
            confidence = predict[7]

            object_mask, noobject_mask, target_x, target_y, target_w, target_h, target_classes, target_confidence, raw_target_w, raw_target_h = build_targets(
                pred_boxes=predict_bboxes,
                pred_cls=torch.sigmoid(classes_probablity),
                target=targets,
                anchors=anchor_vector,
                ignore_thres=self.ignore_thres,
                bbox_loss=self.bbox_loss,
                iou_type=self.iou_type
            )

            loss_bbox = 0
            if self.bbox_loss == 'raw':
                # Loss: 在计算定位损失和类别损失时，使用掩膜过滤掉未匹配上目标的预测框（计算置信度损失不用过滤）
                box_loss_scale = 2.0 - raw_target_w[object_mask] * raw_target_h[object_mask]
                loss_x = (box_loss_scale * self.bce_loss(center_x[object_mask], target_x[object_mask])).mean()
                loss_y = (box_loss_scale * self.bce_loss(center_y[object_mask], target_y[object_mask])).mean()
                loss_w = (0.5 * box_loss_scale * self.mse_loss(width[object_mask], target_w[object_mask])).mean()
                loss_h = (0.5 * box_loss_scale * self.mse_loss(height[object_mask], target_h[object_mask])).mean()
                loss_bbox = loss_x + loss_y + loss_w + loss_h
            elif self.bbox_loss == 'GIoU':
                # 预测坐标取相对于特征图左上角的偏移量，width、height取相对于特征图的像素个数
                center_x, center_y, width, height = predict_bboxes[..., 0], predict_bboxes[..., 1], \
                                                    predict_bboxes[..., 2], predict_bboxes[..., 3]
                grid_x, grid_y = center_x.size()[-2], center_x.size()[-1]
                predict_bboxes_converted = bbox_transfer(center_x[object_mask], center_y[object_mask],
                                                         width[object_mask], height[object_mask], grid_x, grid_y)
                target_bboxes_converted = bbox_transfer(target_x[object_mask], target_y[object_mask],
                                                        target_w[object_mask], target_h[object_mask], grid_x, grid_y)
                loss_bbox = self.giou_loss(predict_bboxes_converted, target_bboxes_converted).mean()

                if loss_bbox.item() < 0:
                    print("\n", loss_bbox)
                    print("bboxes", predict_bboxes_converted)
                    print("target", target_bboxes_converted)

            # 置信度损失
            loss_conf_object = self.bce_loss(confidence[object_mask], target_confidence[object_mask])
            loss_conf_noobject = self.bce_loss(confidence[noobject_mask], target_confidence[noobject_mask])
            loss_conf = self.object_scale * loss_conf_object.mean() + self.noobject_scale * loss_conf_noobject.mean()

            # 类别损失
            loss_cls = self.bce_loss(classes_probablity[object_mask], target_classes[object_mask]).mean()
            current_total_loss = loss_bbox + loss_conf + loss_cls

            loss += current_total_loss
        return loss
