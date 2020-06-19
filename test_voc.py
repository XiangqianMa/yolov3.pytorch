import time, timeit
import numpy as np
import json
import torch
import torchvision.transforms as T
import pickle
import os
from PIL import Image
import tqdm
from models.get_model import GetModel
from utils.postprocess import non_max_suppression
from utils.parse_config import parse_config
from utils.bbox_convert import rescale_boxes, corner_to_upleft
from utils.visualize import visualize
from utils.bbox_convert import pad_to_square, resize
from utils.evaluate_voc import evaluate_detections


class EvaluateVOC(object):
    def __init__(self, model_type, model_cfg, image_size, weight_path, id_to_name_file, cache_path, annopath, setpath, iou_type='iou', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """

        Args:
            model_type: 模型类型
            model_cfg: 模型的配置文件
            image_size: 输入图片的大小
            weight_path: 权重文件的路径
            id_to_name_file: id到类别名的对应关系
            cache_path: 存放中间文件的根目录
            annopath: VOC标注文件存放目录　Annotatioins
            setpath: 存放待评价图片名称的文件路径
            iou_type: iou类型　[iou, giou]
            mean: 通道均值
            std: 通道方差
        """
        self.model_type = model_type
        self.model_cfg = model_cfg
        self.image_size = image_size
        self.weight_path = weight_path
        self.id_to_name = self.__prepare_id_to_name__(id_to_name_file)
        self.classes_name = self.id_to_name.values()
        self.num_classes = len(self.id_to_name)
        self.cache_path = cache_path
        self.iou_type = iou_type
        self.mean = mean
        self.std = std

        self.annopath = annopath
        self.setpath = setpath

        self.__prepare_model__()

    def eval(self, image_paths, conf_thres=0.5, nms_thres=0.5):
        """

        Args:
            image_paths: 样本图片的路径
            conf_thres: 置信度阈值，大于该阈值则被判定为存在目标
            nms_thres: NMS阈值
        """
        predict_boxes = [[[] for _ in range(len(image_paths))] for _ in range(self.num_classes)]
        tbar = tqdm.tqdm(image_paths)
        for image_index, image_path in enumerate(tbar):
            predict = self.detect_single_image(image_path, conf_thres=conf_thres, nms_thres=nms_thres)
            if predict is not None:
                for class_index in range(self.num_classes):
                    current_class_predict = predict[predict[:, -1] == class_index]
                    current_class_predict = current_class_predict[:, :5]
                    predict_boxes[class_index][image_index] = current_class_predict
        
        predict_file = os.path.join(self.cache_path, "predicts.pkl")
        with open(predict_file, 'wb') as f:
            pickle.dump(predict_boxes, f, pickle.HIGHEST_PROTOCOL)
        
        evaluate_detections(predict_boxes, self.cache_path, image_paths, self.classes_name, 
                            'test', self.cache_path, self.annopath, self.setpath)

    def detect_single_image(self, image_path, conf_thres=0.5, nms_thres=0.5):
        """
        检测单张图片
        """
        image, image_tensor = self.__prepare_image__(image_path, self.mean, self.std)
        with torch.no_grad():
            predict = self.model(image_tensor)
            predict = non_max_suppression(predict, conf_thres, nms_thres, iou_type=self.iou_type)[0]

        if predict is not None:
            # 坐标为[x1, y1, x2, y2]形式
            predict = predict.cpu().detach().numpy()
            predict = rescale_boxes(predict, self.image_size, image.shape[:2])

        return predict

    def __prepare_image__(self, image_path, mean, std):
        image = Image.open(image_path).convert("RGB")
        transform_compose = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        image_tensor = transform_compose(image)
        image_tensor, _ = pad_to_square(image_tensor, 0)
        image_tensor = resize(image_tensor, self.image_size)
        image_tensor = torch.unsqueeze(image_tensor, dim=0).cuda()
        image = np.asarray(image)
        return image, image_tensor

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

    def __prepare_id_to_name__(self, id_to_name_file):
        with open(id_to_name_file, 'r') as f:
            id_to_name = json.load(f)
        return id_to_name


if __name__ == "__main__":
    model_type = "darknet"
    model_cfg = "cfg/model_cfg/yolov3-voc.cfg"
    image_size = 416
    iou_type = "iou"
    weight_path = "/home/mxq/Downloads/yolov3_194.pth"
    id_to_name_file = "data/voc/categories_id_to_name.json"
    cache_path = "cache"
    data_root = "data/voc/test"
    annopath = "data/voc/Annotations"
    setpath = "data/voc/ImageSets/Main/test.txt"
    config = parse_config("config.json")
    evaluate_voc = EvaluateVOC(
        model_type,
        model_cfg,
        image_size,
        weight_path,
        id_to_name_file,
        cache_path,
        annopath,
        setpath,
        iou_type=iou_type
    )

    image_paths = sorted([os.path.join(data_root, image_name)  for image_name in os.listdir(data_root)])
    evaluate_voc.eval(image_paths, conf_thres=0.01)
