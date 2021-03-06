#
# 数据增强文件
#
import numpy as np
import random
import cv2
import os
import json
import math
from PIL import Image
from torchvision.transforms import Pad
from albumentations import (
    BboxParams,
    HorizontalFlip,
    VerticalFlip,
    Resize,
    Compose,
    HueSaturationValue,
    ShiftScaleRotate
)

from utils.bbox_convert import center_to_upleft, upleft_to_center
from utils.visualize import visualize


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
            bboxes: list, 对应的目标框, 均为相对于图片宽度和高度的比例形式,[[center_x, center_y, w, h], ...]
            category_id: list, bboxes中各个目标框对应的类别id, [0, 1, ...]
        Return:
            image_augmented: 增强后的图片, Image格式
            bboxes_augmented: list, 增强后的目标框， 均为相对于图片宽度和高度的比例形式,[[center_x, center_y, w, h], ...]
            category_id_augmented: list, 各个目标框对应的类别id， [0, 1, ...]
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
        with open('data/voc/categories_id_to_name.json', 'r') as f:
            categories_id_to_name = json.load(f)
        augmented = self.augmentation(**annotations)
        visualize(augmented, categories_id_to_name)
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
                current_augment = Resize(height=aug_param['height'], width=aug_param['width'], always_apply=aug_param['always_apply'])
            elif aug_name == 'HorizontalFlip':
                current_augment = HorizontalFlip(p=aug_param['p'])
            elif aug_name == 'VerticalFlip':
                current_augment = VerticalFlip(p=aug_param['p'])
            elif aug_name == 'HueSaturationValue':
                current_augment = HueSaturationValue(hue_shift_limit=aug_param['hue_shift_limit'], sat_shift_limit=aug_param['sat_shift_limit'], 
                                                     val_shift_limit=aug_param['val_shift_limit'], p=aug_param['p'])
            elif aug_name == 'ShiftScaleRotate':
                current_augment = ShiftScaleRotate(shift_limit=aug_param['shift_limit'], scale_limit=aug_param['scale_limit'], 
                                                   rotate_limit=aug_param['rotate_limit'], p=aug_param['p'])
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
        boxes: list
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


class MosaicAugment(object):
    """实现mosaic数据增强，将四张图片拼凑在一张图片上
    """
    def __init__(self, image_size, images_list, labels_list):
        self.image_size = image_size
        self.images_list = images_list
        self.labels_list = labels_list

    def load_mosaic(self, index):
        """
        对四张样本进行mosaic增强
        Args:
            index: 必须包含的图片的索引

        Returns:
            img4: mosaic增强后的图片
            labels4: img4对应的标签，labels4[:, 0]为标签，labels[:, 1:]为坐标[x1, y1, x2, y2]
        """
        labels4 = []
        s = self.image_size
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
        img4 = np.zeros((s * 2, s * 2, 3), dtype=np.uint8) + 128  # base image with 4 tiles
        indices = [index] + [random.randint(0, len(self.images_list) - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img = self.load_image(index)
            h, w, _ = img.shape

            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Load labels
            label_path = self.labels_list[index]
            if os.path.isfile(label_path):
                with open(label_path, 'r') as f:
                    x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)

                labels = x.copy()
                if x.size > 0:
                    # Normalized xywh to pixel xyxy format
                    labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                    labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                    labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                    labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh

                labels4.append(labels)
        
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])

        img4, labels4 = random_affine(img4, labels4, degrees=0, translate=0, scale=0, shear=0, border=-s // 2)

        return img4, labels4

    def label_to_yolo_format(self, image, labels):
        """
        将类标转换为yolo所需的格式
        Args:
            image: cv2格式的图片 BGR
            labels: 目标框标注，labels[:, 0]为类标，labels[:, 1:]为目标框
        Returns:
            image: PIL.Image格式 RGB
            category_id: list, 各个bboxes对应的类别id
            converted_bboxes: list, yolo格式的标注框[center_x, center_y, w, h]
        """
        bboxes = labels[:, 1:]
        category_id = labels[:, 0].astype(np.int32)
        converted_bboxes = np.zeros_like(bboxes)
        converted_bboxes[:, 0] = bboxes[:, 0] + (bboxes[:, 2] - bboxes[:, 0]) / 2
        converted_bboxes[:, 1] = bboxes[:, 1] + (bboxes[:, 3] - bboxes[:, 1]) / 2
        converted_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        converted_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        h, _, _ = image.shape
        converted_bboxes = converted_bboxes / h
        converted_bboxes = np.clip(converted_bboxes, 0.0, 1.0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return image, category_id.tolist(), converted_bboxes.tolist()

    def load_image(self, index):
        # loads 1 image from dataset
        img_path = self.images_list[index]
        img = cv2.imread(img_path)  # BGR
        assert img is not None, 'Image Not Found ' + img_path
        r = self.image_size / max(img.shape)  # size ratio
        if r < 1:  # if training (NOT testing), downsize to inference shape
            h, w, _ = img.shape
            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)  # _LINEAR fastest

        return img


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


if __name__ == "__main__":
    images_dir = "data/voc/train"
    labels_dir = "data/voc/train_txt"
    files = os.listdir(images_dir)
    for sample_index in range(30):
        selected = random.choices(files, k=4)
        images_list = [os.path.join(images_dir, a) for a in selected]
        labels_list = [os.path.join(labels_dir, a.replace("jpg", "txt")) for a in selected]

        mosaic_aug = MosaicAugment(416, images_list, labels_list)
        index = random.randint(0, 3)
        image, labels = mosaic_aug.load_mosaic(index)
        bboxes = labels[:, 1:].tolist()
        category_id = labels[:, 0].tolist()
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            bbox[2] = x2 - x1
            bbox[3] = y2 - y1
            bboxes[i] = bbox
            category_id[i] = int(category_id[i])

        annotations = {
            'image': image,
            'bboxes': bboxes,
            'category_id': category_id
        }

        with open('data/voc/categories_id_to_name.json', 'r') as f:
            categories_id_to_name = json.load(f)

        image = visualize(annotations, categories_id_to_name, show=False)
        cv2.imshow("mosaic", image)
        cv2.waitKey(0)
        # cv2.imwrite("masaic_%d.jpg" % sample_index, image)
    pass
