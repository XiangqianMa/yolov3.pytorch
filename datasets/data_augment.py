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
from utils.bbox_convert import center_to_upleft, upleft_to_center


class DataAugment(object):
    def __init__(self, aug=[HorizontalFlip(p=0.5), VerticalFlip(p=0.5),
                            Resize(height=416, width=416, always_apply=True)],
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
            # 多滤掉宽或高为0的目标框
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
        return Compose(self.aug, bbox_params=BboxParams(format=self.dataset_format, min_area=self.min_area,
                                                        min_visibility=self.min_visibility,
                                                        label_fields=['category_id']))


