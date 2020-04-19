import torch
import numpy as np
import torch.nn.functional as F


def upleft_to_center(x_axis, y_axis, width, height, image_width, image_height):
    """将左上角坐标转换为中心点坐标, 并转换为相对于图片宽、高的比例形式

    Args:
        x_axis: x轴坐标
        y_axis: y轴坐标
        width: 宽度
        height: 高度
        image_width: 图片宽度
        image_height: 图片高度
    Return:
        center_x_ratio: 中心坐标x，相对于图片宽度的比例
        center_y_ratio: 中心坐标y，相对于图片高度的比例
        width_ratio: bbox的宽度，相对于图片宽度的比例
        height_ratio: bbox的高度，相对于图片高度的比例
    """
    center_x = x_axis + width / 2
    center_y = y_axis + height / 2

    center_x_ratio = center_x / image_width
    center_y_ratio = center_y / image_height
    width_ratio = width / image_width
    height_ratio = height / image_height

    return center_x_ratio, center_y_ratio, width_ratio, height_ratio


def center_to_upleft(center_x_ratio, center_y_ratio, width_ratio, height_ratio, image_width, image_height):
    """将中心坐标相对于图片宽、高的比例 转换为 左上角坐标

    Args:
        center_x_ratio: 中心x坐标，比例值
        center_y_ratio: 中心y坐标，比例值
        width_ratio: 宽度，比例值
        height_ratio: 高度，比例值
        image_width: 图片宽度
        image_height: 图片高度
    Return:
        left_x: 左上角x坐标
        left_y: 左上角y坐标
        width: 宽度
        height: 高度
    """
    center_x = center_x_ratio * image_width
    center_y = center_y_ratio * image_height

    width = int(width_ratio * image_width)
    height = int(height_ratio * image_height)

    left_x = center_x - width / 2
    left_y = center_y - height / 2

    return left_x, left_y, width, height


def corner_to_upleft(boxes):
    """
    将对角坐标转换为左上角和宽高的形式
    Args:
        boxes: 对角坐标形式的目标框

    Returns:
        boxes_convert: 左上角和宽高形式的坐标
    """
    boxes_convert = np.zeros_like(boxes)
    boxes_convert[..., 0] = boxes[..., 0]
    boxes_convert[..., 1] = boxes[..., 1]
    boxes_convert[..., 2] = boxes[..., 2] - boxes[..., 0]
    boxes_convert[..., 3] = boxes[..., 3] - boxes[..., 1]
    return boxes_convert


def xywh2xyxy(x):
    """
    将中心坐标形式的框转换为对角坐标的形式
    Args:
        x: 中心坐标形式，tensor， [batch_size, boxes_number, 4]

    Returns:
        y: 对角坐标形式，tensor, [batch_size, boxes_number, 4]
    """
    y = x.new_zeros(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2.0
    y[..., 1] = x[..., 1] - x[..., 3] / 2.0
    y[..., 2] = x[..., 0] + x[..., 2] / 2.0
    y[..., 3] = x[..., 1] + x[..., 3] / 2.0
    return y


def rescale_boxes(boxes, current_dim, original_shape):
    """
    将目标框rescale回原始图片对应的尺寸
    Args:
        boxes: 目标框
        current_dim: 当前尺寸
        original_shape: 原始尺寸

    Returns:
        boxes: 原始图片尺寸对应的目标框
    """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image
