import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """YOLO的detection head, 使用网络预测的值和预设的anchor组合得到预测的bbox
    """

    def __init__(self, anchors, number_classes, image_size):
        """
        Args:
            anchors: numpy.array, anchor的尺度大小
            number_classes: 类别数目
            image_size: 图片大小
            arc:
        """
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.number_anchor = len(anchors)
        self.number_classes = number_classes
        self.image_size = image_size

        self.number_grid = None
        self.number_x_grid = 0  # x方向上的grid数量
        self.number_y_grid = 0  # y方向上的grid数量
        self.stride = 0  # 当前yolo layer的步长
        self.grid_xy = None  # grid的偏移量
        self.anchor_wh = None
        self.anchor_vector = None

    def forward(self, feature, image_size):
        """YOLOLayer的前向传播操作

        :param feature: 输入特征图
        :param image_size: 原始输入图片的大小
        :return:
                predict: 所有的预测框，[batch_size, number_of_all_anchors, number_classes+5]
                         number_of_all_anchors = grid_x * grid_y * number_anchors (每一个cell的anchor数目，默认为3)
                output [
                    predict_bboxes,  预测框的坐标（坐标相对于图片左上角，宽高转换为实数）
                    classes_probality, 类别概率, 未sigmoid激活
                    anchor_vector, 使用stride归一化后的anchor
                    center_x, 预测的原始中心点x坐标（相对于cell左上角的偏移值）, 未sigmoid激活
                    center_y, 预测的原始中心点y坐标（相对于cell左上角的偏移值）, 未sigmoid激活
                    width, 宽度, log数据
                    height, 宽度，log数据
                    confidence, 目标置信度, 未sigmoid激活
                    number_x_grid, x方向grid的数量
                    number_y_grid, y方向grid的数量
                    ]
        """
        FloatTensor = torch.cuda.FloatTensor if feature.is_cuda else torch.FloatTensor
        
        batch_size, _, number_y_grid, number_x_grid = feature.size()
        if (self.number_x_grid, self.number_y_grid) != (number_x_grid, number_y_grid):
            self.create_grid(image_size, (number_x_grid, number_y_grid), feature.device, feature.dtype)

        # [batch_size, number_anchor, number_y_grid, number_x_grid, number_classes + 5]
        feature = feature.view(batch_size, self.number_anchor, self.number_classes + 5, self.number_y_grid,
                               self.number_x_grid).permute(0, 1, 3, 4, 2).contiguous()

        # 中心坐标、宽、高
        center_x = feature[..., 0]
        center_y = feature[..., 1]
        width = feature[..., 2]
        height = feature[..., 3]
        # 存在目标的置信度
        confidence = feature[..., 4]
        # 各个类别的概率
        classes_probality = feature[..., 5:]
        # 向预测的bboxes的中心坐标加上偏移，向宽、高乘以尺度
        predict_bboxes = FloatTensor(feature[..., :4].shape)
        predict_bboxes[..., 0] = torch.sigmoid(center_x) + self.grid_xy[..., 0]
        predict_bboxes[..., 1] = torch.sigmoid(center_y) + self.grid_xy[..., 1]
        predict_bboxes[..., 2] = torch.exp(width) * self.anchor_wh[..., 0]
        predict_bboxes[..., 3] = torch.exp(height) * self.anchor_wh[..., 1]

        predict = torch.cat(
            [
                predict_bboxes.view(batch_size, -1, 4) * self.stride,
                torch.sigmoid(confidence).view(batch_size, -1, 1),
                torch.sigmoid(classes_probality).view(batch_size, -1, self.number_classes),
            ], dim=-1,
        )

        if self.training:
            output = [
                predict_bboxes,
                classes_probality,
                self.anchor_vector,
                center_x,
                center_y,
                width,
                height,
                confidence
            ]
        else:
            output = predict

        return output

    def create_grid(self, image_size=416, number_grid=(13, 13), device='cpu', dtype=torch.float32):
        """ 为当前YOLOLayer层创建grid，功能包括：计算每一个grid cell相对于左上角的偏移量，将anchor转换为相对于stride的倍数

        :param image_size: 网络原始输入图像的大小
        :param number_grid: grid的数量, number_grid[0]为x方向，number_grid[1]为y方向
        :param device: cpu / gpu
        :param dtype: 数据类型
        :return:
        """
        number_x_grid, number_y_grid = number_grid
        self.image_size = image_size
        self.stride = self.image_size / max(number_grid)

        # 创建各个grid在x, y坐标轴上的偏移量, 注意x, y轴的方向, x对应特征图的第1维，y对应特征图的第0维
        offset_y, offset_x = torch.meshgrid([torch.arange(number_y_grid), torch.arange(number_x_grid)])
        self.grid_xy = torch.stack([offset_x, offset_y], dim=2).to(device).type(dtype).view(1, 1, number_y_grid,
                                                                                            number_x_grid, 2)

        # 将anchor转换为相对于stride的倍数
        self.anchor_vector = self.anchors.to(device) / self.stride
        self.anchor_wh = self.anchor_vector.view(1, self.number_anchor, 1, 1, 2).to(device).type(dtype)
        self.number_grid = torch.Tensor(number_grid).to(device)
        self.number_x_grid = number_x_grid
        self.number_y_grid = number_y_grid


if __name__ == '__main__':
    val = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
    anchors = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
    yolo_layer = YOLOLayer(anchors, 80, 416)
    feature = torch.zeros([2, 85*9, 13, 13])
    output = yolo_layer(feature, (416, 416))
    print(output[0].size())
