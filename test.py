import numpy as np
import os
import time
import json
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from models.get_model import GetModel
from datasets.coco_dataset import COCODataset
from utils.evaluate import evaluate


class Test(object):
    def __init__(self, model_type, model_cfg, image_size, weight_path, images_root, annotations_root, iou_type='iou'):
        self.model_type = model_type
        self.model_cfg = model_cfg
        self.image_size = image_size
        self.weight_path = weight_path
        self.images_root = images_root
        self.annotations_root = annotations_root
        self.iou_type = iou_type

        self.model = None
        self.__prepare_model__()
        self.dataloader = None
        self.__prepare_data__()

    def __call__(self):
        return evaluate(self.model, self.dataloader, 0.5, 0.001, 0.5, self.image_size, iou_type=self.iou_type)

    def __prepare_data__(self):
        dataset = COCODataset(self.images_root, self.annotations_root, self.image_size, mean=None, std=None)
        self.dataloader = DataLoader(dataset, 2, False, num_workers=8, collate_fn=dataset.collate_fn, pin_memory=True)

    def __prepare_model__(self):
        get_model = GetModel(self.model_type)
        print("@ Creating Model.")
        self.model = get_model.get_model(self.model_cfg, self.image_size)
        print("@ Loading weight from %s." % self.weight_path)
        if self.weight_path.endswith('.pth'):
            state_dict = torch.load(self.weight_path)
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_darknet_weights(self.weight_path)

        self.model = self.model.cuda()
        self.model.eval()


if __name__ == "__main__":
    model_type = "darknet"
    model_cfg = "cfg/model_cfg/yolov3-hand.cfg"
    image_size = 416
    iou_type = "iou"
    weight_path = "checkpoints/backup/log-2020-06-01T23-52-23/weights/yolov3_79.pth"
    images_root = "data/hand/test"
    annotations_root = "data/hand/test_txt"

    test = Test(model_type, model_cfg, image_size, weight_path, images_root, annotations_root, iou_type=iou_type)
    precision, recall, AP, f1, ap_class = test()
    print("Precision: %.4f \n Recall: %.4f \n mAP: %.4f \n f1: %.4f" % (
        precision.mean(), recall.mean(), AP.mean(), f1.mean()))
