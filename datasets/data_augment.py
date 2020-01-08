import json
import numpy as np
from PIL import Image
from albumentations import (
    BboxParams,
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    RandomCrop,
    Crop,
    RandomRotate90,
    Compose
)

from utils.visualize import visualize
from utils.bbox_convert import center_to_upleft


class DataAugment(object):
    def __init__(self, aug=[HorizontalFlip(p=0.5), VerticalFlip(p=0.5),
                            Resize(height=416, width=416, always_apply=True)],
                 dataset_format='coco', min_area=0., min_visibility=0.):

        self.aug = aug
        self.dataset_format = dataset_format
        self.min_area = min_area
        self.min_visibility = min_visibility
        self.augmentation = self.__get_aug__()
    
    def __call__(self, image, bboxes, category_id, image_width, image_height):
        image = np.asarray(image)
        bboxes_converted = []

        category_id_converted = []
        for bbox, current_id in zip(bboxes, category_id):
            center_x, center_y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
            left_x, left_y, width, height = center_to_upleft(center_x, center_y, width, height, image_width, image_height)
            # 多滤掉宽或高为0的目标框
            if width > 0 and height > 0:
                bboxes_converted.append([left_x, left_y, width, height])
                category_id_converted.append(current_id)
        annotations = {
            'image': image,
            'bboxes': bboxes_converted,
            'category_id': category_id_converted
        }

        augmented = self.augmentation(**annotations)
        image = Image.fromarray(augmented['image'])
        bboxes = augmented['bboxes']
        category_id = augmented['category_id']

        return image, bboxes, category_id

    def __get_aug__(self):
        return Compose(self.aug, bbox_params=BboxParams(format=self.dataset_format, min_area=self.min_area,
                                                        min_visibility=self.min_visibility,
                                                        label_fields=['category_id']))


