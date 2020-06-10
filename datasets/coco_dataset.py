#
# 数据集解析文件
# Author: XiangqianMa
#
import torch
import os
import json
import numpy as np
import random
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from datasets.data_augment import DataAugment, pad_to_square, MosaicAugment


class COCODataset(Dataset):
    def __init__(self, images_root, annotations_root, image_size, mean, std, augment=None, normalize=False,
                 multi_scale=False, mosaic=False):
        """
        Args:
            images_root: 存放原始图片的根目录
            annotations_root: 存放标注文件的根目录
            mean: 通道均值
            std: 通道方差
            augment: 图片与bbox的转换方式
            normalize: 是否对图片进行归一化操作(均值为mean，方差为std)
            multi_scale: 是否使用多尺度训练
        """
        self.images_root = images_root
        self.annotations_root = annotations_root
        self.images_list = self.__prepare_images_list__()

        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.augment = augment

        self.multi_scale = multi_scale
        self.max_size = image_size + 32 * 3
        self.min_size = image_size - 32 * 3
        self.batch_count = 0

        self.mosaic = mosaic
        if self.mosaic:
            labels_list = [os.path.join(self.annotations_root, file_name.split('/')[-1].replace('jpg', 'txt'))
                           for file_name in self.images_list]
            self.mosaic_augment = MosaicAugment(self.image_size, self.images_list, labels_list)

    def __getitem__(self, index):
        """
        Return:
            image_path: 图片路径
            image: 样本图片
            categories_id_bboxes: numpy.array, [[sample_id, category_id, x, y, w, h], ...],
                                  sample_id为当前样本在batch中的编号
        """
        image_path = self.images_list[index]
        annotation_path = os.path.join(self.annotations_root, image_path.split('/')[-1].replace('jpg', 'txt'))
        if self.mosaic:
            image, labels = self.mosaic_augment.load_mosaic(index)
            image, categories_id, bboxes = self.mosaic_augment.label_to_yolo_format(image, labels)
        else:
            # 读取与padding
            image = Image.open(image_path).convert("RGB")
            categories_id, bboxes = self.__parse_annotation_txt__(annotation_path)
            image, bboxes = pad_to_square(image, bboxes, fill=0)

        # 普通的数据增强，如翻转等
        if self.augment is not None:
            image, bboxes, categories_id = self.augment(image, bboxes, categories_id)

        # Resize + ToTensor
        compose = [
            T.Resize(self.image_size),
            T.ToTensor()
        ]
        if self.normalize:
            compose.append(T.Normalize(self.mean, self.std))
        transform_compose = T.Compose(compose)
        image = transform_compose(image)

        categories_id_bboxes = None
        if len(bboxes) > 0:
            # categories_id_bboxes[:, 0]用于存放当前样本在batch中的编号
            categories_id_bboxes = torch.zeros((len(bboxes), 6))
            categories_id_bboxes[:, 2:] = torch.Tensor(bboxes)
            categories_id_bboxes[:, 1] = torch.Tensor(categories_id)

        return image_path, image, categories_id_bboxes
    
    def collate_fn(self, batch):
        """将每一个batch中的所有bboxes在第0维进行拼接，其目的是为了方便后续步骤的计算

        Args:
            batch: 
        """
        paths, images, targets = list(zip(*batch))
        # 过滤不存在boxes的样本
        for sample_index, categories_id_bboxes in enumerate(targets):
            if categories_id_bboxes is None:
                continue
            categories_id_bboxes[:, 0] = sample_index
        targets = [categories_id_bboxes for categories_id_bboxes in targets if categories_id_bboxes is not None]
        try:
            targets = torch.cat(targets, dim=0)
        except:
            targets = None

        #  多尺度变换
        if self.multi_scale and self.batch_count % 10 == 0:
            random_image_size = random.choice(range(self.min_size, self.max_size + 1, 32))
            images = [self.__resize__(image, random_image_size) for image in images]
        images = torch.stack(images)
        return paths, images, targets

    def __resize__(self, image, image_size):
        return F.interpolate(image.unsqueeze(0), size=image_size, mode="nearest").squeeze(0)

    def __prepare_images_list__(self):
        annotations_files = os.listdir(self.annotations_root)
        images_list = [os.path.join(self.images_root, annotation_file.replace('txt', 'jpg')) for annotation_file in annotations_files]
        return images_list
    
    def __parse_annotation_txt__(self, txt_path):
        """解析存储标注的txt文件

        Args:
            txt_path: txt文件的路径
        Return:
            categories_id: list, 各个标记框对应的类别, [0, 1, 2, ...]
            bboxes: list, 标记框 [[center_x, center_y, width, height], ...]
        """
        categories_id = []
        bboxes = []
        if os.path.isfile(txt_path):
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
    images_root = 'data/voc/train'
    annotations_root = 'data/voc/train_txt'
    image_size = 416
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    data_augment = DataAugment()
    dataset = COCODataset(images_root, annotations_root, image_size, mean, std, None, False, False, True)
    for i in range(len(dataset)):
        _, image, targets = dataset[i]
    pass
