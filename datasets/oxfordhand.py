import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as A


# boxes = (cls, x, y, w, h)
def horizontal_flip(images, boxes):
    images = np.flip(images, [-1])
    boxes[:, 1] = 1 - boxes[:, 1]
    return images, boxes


# images[np.unit8], boxes[numpy] = (cls, x, y, w, h)
def augment(image, boxes):
    h, w, _ = image.shape
    labels, boxes_coord = boxes[:, 0], boxes[:, 1:]
    labels = labels.tolist()
    boxes_coord = boxes_coord * h     # 得到原图尺寸下的坐标（未归一化的坐标）
    boxes_coord[:, 0] = np.clip(boxes_coord[:, 0]-boxes_coord[:, 2]/2, a_min=0, a_max=None)   # 确保x_min和y_min有效
    boxes_coord[:, 1] = np.clip(boxes_coord[:, 1]-boxes_coord[:, 3]/2, a_min=0, a_max=None)
    boxes_coord = boxes_coord.tolist()      # [x_min, y_min, width, height]

    # 在这里设置数据增强的方法
    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, border_mode=0, p=0.5)
    ], bbox_params={'format':'coco', 'label_fields': ['category_id']})

    augmented = aug(image=image, bboxes=boxes_coord, category_id=labels)

    # 经过aug之后，如果把boxes变没了，则返回原来的图片
    if augmented['bboxes']:
        image = augmented['image']

        boxes_coord = np.array(augmented['bboxes']) # x_min, y_min, w, h → x, y, w, h
        boxes_coord[:, 0] = boxes_coord[:, 0] + boxes_coord[:, 2]/2
        boxes_coord[:, 1] = boxes_coord[:, 1] + boxes_coord[:, 3]/2
        boxes_coord = boxes_coord / h
        labels = np.array(augmented['category_id'])[:, None]
        boxes = np.concatenate((labels, boxes_coord), 1)

    return image, boxes


# 图像的转换流程:PIL→numpy→pad→transform→tensor

# 对numpy格式的img进行padding([0,255])
def pad_to_square(img, pad_value):
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0,0), (0,0)) if h <= w else ((0,0), (pad1, pad2), (0,0))  # 分别对应h,w,c的padding
    # Add padding
    img = np.pad(img, pad, 'constant', constant_values=pad_value)

    return img, (*pad[1], *pad[0])  # 返回w，c的padding


# 对tensor格式的img进行resize
def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]

        img = Image.open(img_path)
        img = np.array(img)

        # Pad to square resolution
        img, _ = pad_to_square(img, 0)

        img = transforms.ToTensor()(img)    # img为np.uint8格式

        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class OxfordDataset(Dataset):
    def __init__(self, image_root, label_root, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        self.image_root = image_root
        self.annotations_root = label_root

        self.img_files, self.label_files = self.__prepare_images_list__()

        self.img_size = img_size
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img[None, :, :]
            img = img.repeat(3, 0)

        h, w, _ = img.shape  # np格式的img是H*W*C
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        padded_h, padded_w, _ = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        assert os.path.exists(label_path)   # 确保label_path必定存在，即图片必定存在label
        boxes = np.loadtxt(label_path).reshape(-1, 5)
        # Extract coordinates for unpadded + unscaled image
        x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        # Adjust for added padding
        x1 += pad[0]    # pad是从低维到高维的，感觉这样写是有问题的，应该只与pad[0][2]有关，不过一般都是相等的
        y1 += pad[2]
        x2 += pad[0]
        y2 += pad[2]
        # Returns (x, y, w, h)
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] *= w_factor / padded_w      # 原来的数值是boxw_ori/imgw_ori, 现在变成了(boxw_ori/imgw_ori)*imgw_ori/imgw_pad=boxw_ori/imgw_pad
        boxes[:, 4] *= h_factor / padded_h

        # Apply augmentations
        # img, 以最长边为标准进行padding得到的uint8图像
        # boxes, (cls, x, y, w, h)都以pad后得到的img的高度进行了归一化
        if self.augment:
            img, boxes = augment(img, boxes)

        img = transforms.ToTensor()(img)   # ToTensor已经将像素值进行了归一化

        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = torch.from_numpy(boxes)  # 0维在collate_fn中是作为idx用了，用于指定target对应的图片

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        # 确保每一张图片都有box，如果某张图片没有标签就会报错！
        for boxes in targets:
            assert (boxes is not None)
        targets = [boxes for boxes in targets if boxes is not None]  # 注意注意！！！这里并没有处理对应的imgs，imgs有可能与targets匹配不上
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        # 因为多线程并行读取的原因，self.batch_count和self.img_size的操作是不对的，
        # 个人觉得更好的处理方法将尺度的变化放到训练阶段，即读取数据之后再做resize
        # if self.multiscale:
        #     self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        if self.multiscale:
            imgs = torch.stack([resize(img, self.max_size) for img in imgs])    # 先将img统一resize到最大值
        else:
            imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

    # 重新设置img_size
    def select_new_img_size(self):
        self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

    # 对tensor格式的img进行resize
    def resize_imgs(self, images):
        if self.multiscale:
            images = F.interpolate(images, size=self.img_size, mode="nearest")

        return images

    def __prepare_images_list__(self):
        annotations_files = os.listdir(self.annotations_root)
        images_list = [os.path.join(self.image_root, annotation_file.replace('txt', 'jpg')) for annotation_file in annotations_files]
        annotations_list = [os.path.join(self.annotations_root, annotation_file) for annotation_file in annotations_files]
        return images_list, annotations_list
