import numpy as np
import torch
from utils.bbox_convert import xywh2xyxy


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    """

    Args:
        prediction: 网络的预测输出，[batch_size, boxes_number, 4 + 1 + classes_number]
        conf_thres: 是否存在目标的置信度的阈值
        nms_thres: 目标框重合的阈值

    Returns:

    """
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # 将边界框从中心坐标的形式转换为对角坐标
    outputs = [None for _ in range(len(prediction))]
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
            # TODO


if __name__ == '__main__':
    prediction = torch.empty(5, 6, 20)
    output = non_max_suppression(prediction, 0.5, 0.5)
    pass

