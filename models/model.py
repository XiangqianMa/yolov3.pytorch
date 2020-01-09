import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.layers import YOLOLayer
from models.parse_model_cfg import parse_model_cfg
from datasets.coco_dataset import COCODataset
from datasets.data_augment import DataAugment
from losses.yolo_loss import YOLOLoss


class Darknet(nn.Module):
    # YOLOv3 object detection models

    def __init__(self, cfg, img_size=(416, 416)):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training

    def forward(self, x):
        """

        Return:
            训练时，返回：[[predict_bboxes, classes_probality, anchor_vector], ..], 用于计算损失
            测试时，返回：[batch_size, number_all_anchors, number_classes+5], 用于得到最终的预测结果
        """
        img_size = x.shape[-2:]
        layer_outputs = []
        output = []

        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    try:
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    except:
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
            elif mtype == 'shortcut':
                x = x + layer_outputs[int(mdef['from'])]
            elif mtype == 'yolo':
                x = module(x, img_size)
                output.append(x)
            layer_outputs.append(x if i in self.routs else [])
        if not self.training:
            output = torch.cat(output, dim=1)
        return output


def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']  # [82, 94, 106] for yolov3


def create_modules(module_defs, img_size):
    """依据module_defs中的网络定义构建整个网络结构

    :param module_defs: 网络模块的定义
    :param img_size: 图像大小
    :return: module_list: 网络模块, routs: 路由到更深的层的层列表
    """

    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            size = int(mdef['size'])
            stride = int(mdef['stride']) if 'stride' in mdef else (int(mdef['stride_y']), int(mdef['stride_x']))
            pad = (size - 1) // 2 if int(mdef['pad']) else 0
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=size,
                                                   stride=stride,
                                                   padding=pad,
                                                   groups=int(mdef['groups']) if 'groups' in mdef else 1,
                                                   bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            if mdef['activation'] == 'leaky':  # TODO: activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))

        elif mdef['type'] == 'maxpool':
            size = int(mdef['size'])
            stride = int(mdef['stride'])
            maxpool = nn.MaxPool2d(kernel_size=size, stride=stride, padding=int((size - 1) // 2))
            if size == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'upsample':
            modules = nn.Upsample(scale_factor=int(mdef['stride']), mode='nearest')

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) for x in mdef['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            routs.extend([l if l > 0 else l + i for l in layers])

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            filters = output_filters[int(mdef['from'])]
            layer = int(mdef['from'])
            routs.extend([i + layer if layer < 0 else layer])

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            mask = [int(x) for x in mdef['mask'].split(',')]  # anchor mask
            modules = YOLOLayer(
                anchors=mdef['anchors'][mask],
                number_classes=int(mdef['classes']),
                image_size=img_size,
                )  # yolo architecture
        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list, routs


if __name__ == '__main__':
    model_cfg_path = 'cfg/model_cfg/yolov3.cfg'
    model = Darknet(model_cfg_path, (416, 416))
    images_root = 'data/coco/val2017'
    annotations_root = 'data/coco/val2017_txt'
    image_size = 416
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    data_augment = DataAugment()
    dataset = COCODataset(images_root, annotations_root, image_size, mean, std, data_augment)
    dataloader = DataLoader(dataset, 8, True, num_workers=8, pin_memory=True, collate_fn=dataset.collate_fn)
    criterion = YOLOLoss(0.5)
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    for _, images, targets in dataloader:
        output = model(images)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss.item())
    pass

