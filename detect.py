import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from models.get_model import GetModel
from utils.postprocess import non_max_suppression
from utils.parse_config import parse_config
from utils.bbox_convert import rescale_boxes


class Detect(object):
    def __init__(self, model_type, model_cfg, image_size, weight_path):
        self.model_type = model_type
        self.model_cfg = model_cfg
        self.image_size = image_size
        self.weight_path = weight_path

        self._prepare_model()

    def detect_single_image(self, image_path, mean, std, conf_thres=0.5, nms_thres=0.5):
        image, image_tensor = self._prepare_image(image_path, mean, std)
        with torch.no_grad():
            predict = self.model(image_tensor)
            predict = non_max_suppression(predict, conf_thres, nms_thres)[0]
        predict = predict.cpu().detach().numpy()
        predict = rescale_boxes(predict, [self.image_size, self.image_size], image.shape[:2])
        # TODO 将检测结果可视化
        pass

    def detect_multi_images(self, image_dir, mean, std, conf_thres=0.5, nms_thres=0.5):
        pass

    def _prepare_image(self, image_path, mean, std):
        image = Image.open(image_path).convert("RGB")
        transform_compose = T.Compose(
            [
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        image_tensor = transform_compose(image)
        image_tensor = torch.unsqueeze(image_tensor, dim=0).cuda()
        image = np.asarray(image)
        return image, image_tensor

    def _prepare_model(self):
        get_model = GetModel(self.model_type)
        print("@ Creating Model.")
        self.model = get_model.get_model(self.model_cfg, self.image_size)
        print("@ Loading weight from %s." % self.weight_path)
        state_dict = torch.load(self.weight_path)
        self.model.load_state_dict(state_dict)
        self.model = self.model.cuda()
        self.model.eval()


if __name__ == "__main__":
    weight_path = "checkpoints/backup/log-2020-04-16T16-57-00/weights/yolov3_79.pth"
    image_path = "data/test_images/000000581781.jpg"
    config = parse_config("config.json")
    detect = Detect(
        config["model_type"],
        config["model_cfg"],
        config["image_size"],
        weight_path
    )

    detect.detect_single_image(
        image_path,
        config["mean"],
        config["std"],
        0.5,
        0.5
    )
    pass

