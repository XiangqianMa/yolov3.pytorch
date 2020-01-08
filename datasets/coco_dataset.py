import torch
import os
import json
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt

from datasets.data_augment import DataAugment


class COCODataset(Dataset):
    def __init__(self, images_root, annotations_root, image_size, mean, std, transforms=None):
        """
        Args:
            images_root: 存放原始图片的根目录
            annotations_root: 存放标注文件的根目录
            image_size: resize后的图片大小
            mean: 通道均值
            std: 通道方差
            transforms: 图片与bbox的转换方式
        """
        self.images_root = images_root
        self.annotations_root = annotations_root
        self.images_list = self.__prepare_images_list__()
        
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.images_list[index]
        annotation_path = os.path.join(self.annotations_root, image_path.split('/')[-1].replace('jpg', 'txt'))
        image = Image.open(image_path).convert("RGB")
        categories_id, bboxes = self.__parse_annotation_txt__(annotation_path)
        image, bboxes, categories_id = self.transforms(image, bboxes, categories_id, image.size[0], image.size[1])

        transform_compose = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        image = transform_compose(image)

        return image, bboxes, categories_id

    def __prepare_images_list__(self):
        annotations_files = os.listdir(self.annotations_root)
        images_list = [os.path.join(self.images_root, annotation_file.replace('txt', 'jpg')) for annotation_file in annotations_files]
        return images_list
    
    def __parse_annotation_txt__(self, txt_path):
        """解析存储标注的txt文件

        Args:
            txt_path: txt文件的路径
        Return:
            categories_id: 各个标记框对应的类别, [0, 1, 2, ...]
            bboxes: 标记框 [[center_x, center_y, width, height], ...]
        """
        categories_id = []
        bboxes = []
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                annotation_line = line.split(' ')
                category_id, center_x, center_y, width, height = int(annotation_line[0]), float(annotation_line[1]), float(annotation_line[2]), float(annotation_line[3]), float(annotation_line[4])
                categories_id.append(category_id)
                bboxes.append([center_x, center_y, width, height])
        return categories_id, bboxes

    def __len__(self):
        return len(self.images_list)


if __name__ == '__main__':
    images_root = 'data/coco/val2017'
    annotations_root = 'data/coco/val2017_txt'
    image_size = 416
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    data_augment = DataAugment()
    dataset = COCODataset(images_root, annotations_root, image_size, mean, std, data_augment)
    for i in range(len(dataset)):
        image, bboxes, categories_id = dataset[i]
    pass
