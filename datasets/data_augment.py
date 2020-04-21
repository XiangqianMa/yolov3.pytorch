#
# 数据增强文件
#
import numpy as np
from PIL import Image
from torchvision.transforms import Pad
from albumentations import (
    BboxParams,
    HorizontalFlip,
    VerticalFlip,
    Resize,
    Compose
)

from utils.bbox_convert import center_to_upleft, upleft_to_center


class DataAugment(object):
    def __init__(self,
                 aug={'Resize': {'height': 416, 'width': 416, 'always_apply': True}},
                 dataset_format='coco', min_area=0., min_visibility=0.):

        self.aug = aug
        self.dataset_format = dataset_format
        self.min_area = min_area
        self.min_visibility = min_visibility
        self.augmentation = self.__get_aug__()
    
    def __call__(self, image, bboxes, category_id):
        """对输入的样本图片和bboxes进行增强

        Args:
            image: 样本图片, Image格式
            bboxes: 对应的目标框, 均为相对于图片宽度和高度的比例形式,[[center_x, center_y, w, h], ...]
            category_id: bboxes中各个目标框对应的类别id, [0, 1, ...]
        Return:
            image_augmented: 增强后的图片, Image格式
            bboxes_augmented: 增强后的目标框， 均为相对于图片宽度和高度的比例形式,[[center_x, center_y, w, h], ...]
            category_id_augmented: 各个目标框对应的类别id， [0, 1, ...]
        """
        image_width = image.size[0]
        image_height = image.size[1]
        image = np.asarray(image)
        bboxes_converted = []

        category_id_converted = []
        for bbox, current_id in zip(bboxes, category_id):
            center_x, center_y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
            left_x, left_y, width, height = center_to_upleft(center_x, center_y, width, height, image_width, image_height)
            # 过滤掉宽或高为0的目标框
            if width > 0 and height > 0:
                bboxes_converted.append([left_x, left_y, width, height])
                category_id_converted.append(current_id)
        annotations = {
            'image': image,
            'bboxes': bboxes_converted,
            'category_id': category_id_converted
        }
        # with open('data/coco/categories_id_to_name.json', 'r') as f:
        #     categories_id_to_name = json.load(f)
        augmented = self.augmentation(**annotations)
        # visualize(augmented, categories_id_to_name)
        image_augmented = Image.fromarray(augmented['image'])
        bboxes_augmented = augmented['bboxes']
        for bbox_index, bbox_agumented in enumerate(bboxes_augmented):
            center_x_augmented, center_y_augmented, width_augmented, height_augmented = upleft_to_center(
                 bbox_agumented[0], bbox_agumented[1], 
                 bbox_agumented[2], bbox_agumented[3], 
                 image_augmented.size[0], image_augmented.size[1])
            bboxes_augmented[bbox_index] = [center_x_augmented, center_y_augmented, width_augmented, height_augmented]

        category_id_augmented = augmented['category_id']

        return image_augmented, bboxes_augmented, category_id_augmented

    def __get_aug__(self):
        augment = []
        for aug_name, aug_param in self.aug.items():
            current_augment = []
            if aug_name == 'Resize':
                assert ('height' in aug_param and 'width' in aug_param and 'always_apply' in aug_param)
                current_augment = Resize(height=aug_param['height'], width=aug_param['width'], always_apply=aug_param['always_apply'])
            elif aug_name == 'HorizontalFlip':
                assert ('p' in aug_param)
                current_augment = HorizontalFlip(p=aug_param['p'])
            elif aug_name == 'VerticalFlip':
                assert ('p' in aug_param)
                current_augment = VerticalFlip(p=aug_param['p'])
            augment.append(current_augment)
        
        return Compose(augment, bbox_params=BboxParams(format=self.dataset_format, min_area=self.min_area,
                                                        min_visibility=self.min_visibility,
                                                        label_fields=['category_id']))


def pad_to_square(image, boxes, fill=0):
    """
    对输入图片进行pad操作，将较短的边补为和长边一样的大小
    Args:
        image: PIL.Image
        boxes: list，目标的标注框
        fill: pad的值，默认为0

    Returns:
        image: PIL.Image
        boxes: List
    """
    w, h = image.size
    size_diff = np.abs(w - h)
    pad1 = size_diff // 2
    pad2 = size_diff - pad1
    # (left, top, right, bottom)
    pad_diff = (0, pad1, 0, pad2) if h <= w else (pad1, 0, pad2, 0)
    pad = Pad(pad_diff, fill=fill)
    image = pad(image)
    padded_w, padded_h = image.size
    boxes_numpy = np.asarray(boxes)
    # [center_x, center_y, w, h] -> [x1, y1, x2, y2]
    x1 = (boxes_numpy[:, 0] - boxes_numpy[:, 2] / 2) * w
    y1 = (boxes_numpy[:, 1] - boxes_numpy[:, 3] / 2) * h
    x2 = (boxes_numpy[:, 0] + boxes_numpy[:, 2] / 2) * w
    y2 = (boxes_numpy[:, 1] + boxes_numpy[:, 3] / 2) * h
    # padding操作
    x1 += pad_diff[0]
    y1 += pad_diff[1]
    x2 += pad_diff[0]
    y2 += pad_diff[1]
    # [x1, y1, x2, y2] -> [center_x, center_y, w, h]
    boxes_numpy[:, 0] = ((x1 + x2) / 2) / padded_w
    boxes_numpy[:, 1] = ((y1 + y2) / 2) / padded_h
    boxes_numpy[:, 2] = (x2 - x1) / padded_w
    boxes_numpy[:, 3] = (y2 - y1) / padded_h
    boxes = boxes_numpy.tolist()
    return image, boxes
