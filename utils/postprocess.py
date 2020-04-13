import numpy as np
import torch
from utils.bbox_convert import xywh2xyxy
from utils.calculate_iou import bbox_iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    """分类别对预测框进行NMS处理，在去除重复的目标框时采取的方法是将大于nms_thres的目标框加权到该类别得分最高的目标框的方式。
       权重为各目标框存在目标的置信度

    Args:
        prediction: 网络的预测输出，[batch_size, boxes_number, 4 + 1 + classes_number]
        conf_thres: 是否存在目标的置信度的阈值
        nms_thres: 目标框重合的阈值

    Returns:
        output: 经过NMS处理后剩余的目标框，大小为[batch_size, boxes_number, 4 + 1 + 2]，
                元素代表的含义依次为：目标框的（x1,y1, x2, y2）坐标、存在目标的置信度、类别分数、类别编号
    """
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # 将边界框从中心坐标的形式转换为对角坐标
    output = [None for _ in range(len(prediction))]
    # 遍历每张图片
    for image_index, image_prediction in enumerate(prediction):
        # 过滤预测框中置信度低于阈值的框
        image_prediction = image_prediction[image_prediction[..., 4] >= conf_thres]
        if not image_prediction.size(0):
            continue

        # 每个预测框的最终得分：confidence * 其所属的类别的分数，即P(class|exist_object)
        score = image_prediction[..., 4] * image_prediction[..., 5:].max(1)[0]
        # 按照得分对预测框从大到小进行排序
        image_prediction = image_prediction[(-score).argsort()]
        class_confidence, class_preds = image_prediction[..., 5:].max(1, keepdim=True)  # 预测的类别得分以及类别编号
        detections = torch.cat((image_prediction[..., :5], class_confidence.float(), class_preds.float()), dim=1)
        # 进行非极大抑制处理
        keep_boxes = []
        while detections.size(0):
            # 计算分数最高的bbox与其余bboxs的iou，并使用nms_thres进行过滤
            lager_overlap = bbox_iou(detections[0, :4].unsqueeze(dim=0), detections[:, :4]) > nms_thres
            same_class = detections[:, -1] == detections[0, -1]
            # 过滤掉属于同一类别的，且与最高分数的目标框的IoU大于nms_thres的目标框
            invalid_index = lager_overlap & same_class
            # 将这些框按照存在目标的置信度为权重加权到得分最高的框中
            weights = detections[invalid_index, 4:5]
            detections[0, :4] = (weights * detections[invalid_index, :4]).sum(dim=0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid_index]  # 过滤掉当前最高分数的框 以及 无效的框

        if keep_boxes:
            output[image_index] = torch.stack(keep_boxes)

        return output


if __name__ == '__main__':
    prediction = torch.ones(5, 6, 20)
    output = non_max_suppression(prediction, 0.5, 0.5)
    pass

