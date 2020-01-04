import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class YOLOLayer(nn.Module):
    """YOLO的detection head, 使用网络预测的值和预设的anchor组合得到预测的bbox
    """

    def __init__(self, anchors, number_classes, image_size, yolo_index, arc):
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
        self.arc = arc

        self.number_grid = None
        self.number_x_grid = 0  # x方向上的grid数量
        self.number_y_grid = 0  # y方向上的grid数量
        self.stride = 0  # 当前yolo layer的步长
        self.grid_xy = None  # grid的偏移量
        self.anchor_wh = None
        self.anchor_vector = None

    def forward(self, feature, image_size, var=None):
        """YOLOLayer的前向传播操作

        :param feature: 输入特征图
        :param image_size: 原始输入图片的大小
        :param var:
        :return: 训练时返回 [batch_size, number_anchor, number_y_grid, number_x_grid, number_classes + 5]
                 测试时返回 [batch_size, all_anchors, number_classes + 5]
        """
        batch_size, _, number_y_grid, number_x_grid = feature.size()
        if (self.number_x_grid, self.number_y_grid) != (number_x_grid, number_y_grid):
            self.create_grid(image_size, (number_x_grid, number_y_grid), feature.device, feature.dtype)

        # [batch_size, number_anchor, number_y_grid, number_x_grid, number_classes + 5]
        feature = feature.view(batch_size, self.number_anchor, self.number_classes + 5, self.number_y_grid,
                               self.number_x_grid).permute(0, 1, 3, 4, 2).contiguous()
        if self.training:
            return feature
        else:
            inference_output = feature.clone()
            inference_output[..., 0:2] = torch.sigmoid(inference_output[..., 0:2]) + self.grid_xy  # x, y
            inference_output[..., 2:4] = torch.exp(inference_output[..., 2:4]) * self.anchor_wh  # w, h
            inference_output[..., :4] = inference_output[..., :4] * self.stride

        if 'default' in self.arc:
            inference_output[..., 4:] = torch.sigmoid(inference_output[..., 4:])
        elif 'BCE' in self.arc:
            inference_output[..., 5:] = torch.sigmoid(inference_output[..., 5:])
            inference_output[..., 4] = 1
        elif 'CE' in self.arc:
            inference_output[..., 5:] = torch.softmax(inference_output[..., 5:], dim=4)
            inference_output[..., 4] = 1

        if self.number_classes == 1:
            inference_output[..., 5] = 1
        return inference_output.view(batch_size, -1, 5 + self.number_classes), feature

    def create_grid(self, image_size=(416, 416), number_grid=(13, 13), device='cpu', dtype=torch.float32):
        """ 为当前YOLOLayer层创建grid，功能包括：计算每一个grid cell相对于左上角的偏移量，将anchor转换为相对于stride的倍数

        :param image_size: 网络原始输入图像的大小
        :param number_grid: grid的数量, number_grid[0]为x方向，number_grid[1]为y方向
        :param device: cpu / gpu
        :param dtype: 数据类型
        :return:
        """
        number_x_grid, number_y_grid = number_grid
        self.image_size = max(image_size)
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


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x.mul_(torch.sigmoid(x))


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x.mul_(F.softplus(x).tanh())


if __name__ == '__main__':
    val = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
    anchors = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
    yolo_layer = YOLOLayer(anchors, 80, 416, 'default')
    feature = torch.zeros([2, 85*9, 13, 13])
    yolo_layer.eval()
    output = yolo_layer(feature, (416, 416))
    print(output[0].size())
