#
# 构造损失函数
#
import torch
from torch.nn import Module
from losses.yolo_loss import YOLOLoss


class GetLoss(Module):
    def __init__(
        self, 
        loss_type={'yolo_loss': {'object_scale': 1, 'noobject_scale': 100, 'ignore_thresh': 0.5}}, 
        loss_weights=[1.0]
    ):
        super(GetLoss, self).__init__()
        self.loss_type = loss_type
        self.loss_weights = loss_weights
        self.losses = self.__get_loss__()
        self.__logger__()

    def forward(self, predicts, targets):
        loss = 0
        for weight, loss_func in zip(self.loss_weights, self.losses):
            loss += weight * loss_func(predicts, targets)
        
        return loss

    def __get_loss__(self):
        losses = torch.nn.ModuleList()
        for loss_name, param in self.loss_type.items():
            if loss_name == 'yolo_loss':
                assert ('object_scale' in param.keys() and 'noobject_scale' in param.keys()
                        and 'ignore_thresh' in param.keys() and 'bbox_loss' in param.keys())
                losses.append(YOLOLoss(ignore_thres=param['ignore_thresh'], object_scale=param['object_scale'],
                                       noobject_scale=param['noobject_scale'], bbox_loss=param['bbox_loss']))
            else:
                raise NotImplementedError
        return losses
    
    def __logger__(self):
        descript = '@ Losses: '
        for index, [weight, loss] in enumerate(zip(self.loss_weights, self.loss_type.keys())):
            if index < len(self.loss_weights) - 1:
                descript += str(weight) + ' * ' + loss + ' + '
            else:
                descript += str(weight) + ' * ' + loss
        print(descript)
