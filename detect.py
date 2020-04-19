import numpy as np
import os
import json
import torch
import torchvision.transforms as T
from PIL import Image
from models.get_model import GetModel
from utils.postprocess import non_max_suppression
from utils.parse_config import parse_config
from utils.bbox_convert import rescale_boxes, corner_to_upleft
from tools.visualize import visualize
from utils.bbox_convert import pad_to_square, resize


class Detect(object):
    def __init__(self, model_type, model_cfg, image_size, weight_path, id_to_name_file, save_path):
        self.model_type = model_type
        self.model_cfg = model_cfg
        self.image_size = image_size
        self.weight_path = weight_path
        self.id_to_name = self._prepare_id_to_name(id_to_name_file)
        self.save_path = save_path

        self._prepare_model()

    def detect_single_image(self, image_path, mean, std, conf_thres=0.5, nms_thres=0.5):
        image, image_tensor = self._prepare_image(image_path, mean, std)
        with torch.no_grad():
            predict = self.model(image_tensor)
            predict = non_max_suppression(predict, conf_thres, nms_thres)[0]
        # 坐标为[x1, y1, x2, y2]形式
        predict = predict.cpu().detach().numpy()
        predict = rescale_boxes(predict, self.image_size, image.shape[:2])
        # [x1, y1, x2, y2] -> [x1, y1, w, h]
        predict[..., :4] = corner_to_upleft(predict[..., :4])
        predict_boxes = predict[..., :4].tolist()
        predict_id = predict[..., -1].astype(int).tolist()
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
        pass

    def detect_multi_images(self, image_dir, mean, std, conf_thres=0.5, nms_thres=0.5):
        pass

    def _prepare_image(self, image_path, mean, std):
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

    def _prepare_model(self):
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

    def _prepare_id_to_name(self, id_to_name_file):
        with open(id_to_name_file, 'r') as f:
            id_to_name = json.load(f)
        return id_to_name


if __name__ == "__main__":
    weight_path = "checkpoints/official_weights/yolov3.weights"
    image_path = "data/test_images/000000256868.jpg"
    id_to_name_file = "data/coco/categories_id_to_name.json"
    save_path = "data/test_results"
    config = parse_config("config.json")
    detect = Detect(
        config["model_type"],
        config["model_cfg"],
        config["image_size"],
        weight_path,
        id_to_name_file,
        save_path
    )

    detect.detect_single_image(
        image_path,
        config["mean"],
        config["std"],
        0.5,
        0.5
    )
    pass

