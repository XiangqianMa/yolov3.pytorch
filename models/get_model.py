from models.darknet import Darknet
from models.darknet import load_darknet_weights


class GetModel(object):
    def __init__(self, model_type):
        self.model_type = model_type
        self.__logger__()

    def get_model(self, cfg={'cfg': '', 'image_size': 416}, pretrained_weight=None):
        if self.model_type == 'darknet':
            assert ('cfg' in cfg.keys() and 'image_size' in cfg.keys())
            model = Darknet(cfg['cfg'], cfg['image_size'])
        else:
            raise NotImplementedError
        
        if pretrained_weight is not None:
            print('@ Loading from %s' % pretrained_weight)
            load_darknet_weights(model, pretrained_weight)

        return model
    
    def __logger__(self):
        print('@ Model Type: %s' % self.model_type)


if __name__ == '__main__':
    cfg = ''
    get_model = GetModel('darknet')
    model = get_model.get_model(cfg={'cfg': 'cfg/model_cfg/yolov3.cfg', 'image_size': 416})
    pass