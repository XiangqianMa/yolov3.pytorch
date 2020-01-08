

def upleft_to_center(x_axis, y_axis, width, height, image_width, image_height):
    """将左上角坐标转换为中心点坐标, 并转换为相对于图片宽、高的比例形式

    Args:
        x_axis: x轴坐标
        y_axis: y轴坐标
        width: 宽度
        height: 高度
        image_widht: 图片宽度
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