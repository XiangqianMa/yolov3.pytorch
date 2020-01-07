

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

