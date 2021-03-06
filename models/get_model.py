import torch
from models.darknet import Darknet


class GetModel(object):
    def __init__(self, model_type):
        self.model_type = model_type
        self.__logger__()

    def get_model(self, model_cfg, image_size, pretrained_weight=None):
        if self.model_type == 'darknet':
            model = Darknet(model_cfg, image_size)
        else:
            raise NotImplementedError
        
        if pretrained_weight is not None:
            print('@ Loading from %s' % pretrained_weight)
            if pretrained_weight.endswith('.pth'):
                state_dict = torch.load(pretrained_weight)
                model.load_state_dict(state_dict)
            else:
                model.load_darknet_weights(pretrained_weight)

        return model
    
    def __logger__(self):
        print('@ Model Type: %s' % self.model_type)


if __name__ == '__main__':
    cfg = ''
    get_model = GetModel('darknet')
    model = get_model.get_model(cfg={'cfg': 'cfg/model_cfg/yolov3.cfg', 'image_size': 416})
    pass