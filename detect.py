import numpy as np
import os
import time
import json
import torch
import torchvision.transforms as T
from PIL import Image
from models.get_model import GetModel
from utils.postprocess import non_max_suppression
from utils.parse_config import parse_config
from utils.bbox_convert import rescale_boxes, corner_to_upleft
from utils.visualize import visualize
from utils.bbox_convert import pad_to_square, resize


class Detect(object):
    def __init__(self, model_type, model_cfg, image_size, weight_path, id_to_name_file, save_path, iou_type='iou'):
        self.model_type = model_type
        self.model_cfg = model_cfg
        self.image_size = image_size
        self.weight_path = weight_path
        self.id_to_name = self.__prepare_id_to_name__(id_to_name_file)
        self.save_path = save_path
        self.iou_type = iou_type

        self.__prepare_model__()

    def detect_single_image(self, image_path, mean, std, conf_thres=0.5, nms_thres=0.5):
        print('~' * 10 + image_path.split('/')[-1] + '~' * 10)
        image, image_tensor = self.__prepare_image__(image_path, mean, std)
        with torch.no_grad():
            start_time = time.time()
            predict = self.model(image_tensor)
            end_time = time.time()
            print("@ Inference and Boxes Analysis took %d ms." % ((end_time - start_time) * 1000))
            start_time = time.time()
            predict = non_max_suppression(predict, conf_thres, nms_thres, iou_type=self.iou_type)[0]
            end_time = time.time()
            print("@ NMS took %d ms." % ((end_time - start_time) * 1000))

        if predict is not None:
            # 坐标为[x1, y1, x2, y2]形式
            predict = predict.cpu().detach().numpy()
            predict = rescale_boxes(predict, self.image_size, image.shape[:2])
            # [x1, y1, x2, y2] -> [x1, y1, w, h]
            predict[..., :4] = corner_to_upleft(predict[..., :4])
            predict_boxes = predict[..., :4].tolist()
            predict_conf = predict[..., 4].tolist()
            predict_id = predict[..., -1].astype(int).tolist()
            self.__log_predicts__(predict_conf, predict_id)
            annotations = {
                'image': image,
                'bboxes': predict_boxes,
                'category_id': predict_id,
            }
            image_with_bboxes = visualize(annotations, self.id_to_name)
            image_with_bboxes = Image.fromarray(image_with_bboxes)
            image_save_path = os.path.join(self.save_path, image_path.split('/')[-1])
            print("@ Saving image to %s." % image_save_path)
            image_with_bboxes.save(image_save_path)
        else:
            print("@ No object in %s." % image_path)
        print("\n")
        pass

    def detect_multi_images(self, image_dir, mean, std, conf_thres=0.5, nms_thres=0.5):
        # 序列化检测的方式，非batch
        images_list = os.listdir(image_dir)
        for image_name in images_list:
            image_path = os.path.join(image_dir, image_name)
            self.detect_single_image(image_path, mean, std, conf_thres, nms_thres)
        pass

    def __log_predicts__(self, predict_conf, predict_id):
        for index, (conf, class_id) in enumerate(zip(predict_conf, predict_id)):
            print("  >> Object_%d: %s - %.4f." % (index, self.id_to_name[str(class_id)], conf))

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
    model_cfg = "cfg/model_cfg/yolov3-hand.cfg"
    image_size = 416
    iou_type = "iou"
    weight_path = "checkpoints/backup/log-2020-06-01T23-52-23/weights/yolov3_79.pth"
    image_root = "data/test_images"
    image_path = "data/test_images/000000217060.jpg"
    id_to_name_file = "data/coco/categories_id_to_name.json"
    save_path = "data/test_results"
    config = parse_config("config.json")
    detect = Detect(
        model_type,
        model_cfg,
        image_size,
        weight_path,
        id_to_name_file,
        save_path,
        iou_type=iou_type
    )

    detect.detect_multi_images(image_root, config["mean"], config["std"], 0.5, 0.5)

    # detect.detect_single_image(
    #     image_path,
    #     config["mean"],
    #     config["std"],
    #     0.5,
    #     0.5
    # )
    pass

