import torch.optim as optim
from torch.optim import lr_scheduler
from models.get_model import GetModel
from losses.get_loss import GetLoss
from datasets.get_dataloader import GetDataLoader


class Prepare(object):
    """准备模型和优化器
    """

    def __init__(self):
        pass

    def create_model(self, model_type, model_cfg, image_size, pretrained_weight=None):
        """创建模型
        Args:
            model_type: str, 模型类型
            cfg: 网络结构配置
            pretrained_weight: 预训练权重的路径
        """
        print('Creating model: {}'.format(model_type))
        model = GetModel(model_type).get_model(model_cfg=model_cfg, image_size=image_size, pretrained_weight=pretrained_weight)

        return model

    def create_dataloader(self, config):
        my_get_dataloader = GetDataLoader(
            config["train_images_root"],
            config["train_annotations_root"],
            config["val_images_root"],
            config["val_annotations_root"],
            config["mean"],
            config["std"],
            config["dataset_format"],
            config["train_augmentation"],
            config["val_augmentation"]
            )
        train_dataloader, val_dataloader = my_get_dataloader.get_dataloader(config["batch_size"])
        return train_dataloader, val_dataloader

    def create_criterion(self, loss_type, loss_weights):
        criterion = GetLoss(loss_type=loss_type, loss_weights=loss_weights)

        return criterion

    def create_optimizer(self, model, config):
        """返回优化器

        Args:
            model: 待优化的模型
            config: 配置
        Return:
            optimizer: 优化器
        """
        print('Creating optimizer: %s' % config["optimizer"])
        if config["optimizer"] == 'Adam':
            optimizer = optim.Adam(
                [
                    {'params': model.module.parameters(), 'lr': config["lr"]}
                ], weight_decay=config["weight_decay"])
        elif config["optimizer"] == 'SGD':
            optimizer = optim.SGD(
                [
                    {'params': model.module.parameters(), 'lr': config["lr"]}
                ], weight_decay=config["weight_decay"], momentum=0.9)
        else:
            raise NotImplementedError("Please supply a right optimizer type.")

        return optimizer

    def create_lr_scheduler(self, lr_scheduler_type, optimizer, step_size=None, restart_step=None, multi_step=None):
        """创建学习率衰减器
        Args:
            lr_scheduler_type: 衰减器类型
            optimizer: 优化器
            step_size: 使用StepLR时，必须指定该参数
            restart_step: CosineLR中使用的重启步长
            multi_step: MultiStep中的multi_step
        Return:
            my_lr_scheduler: 学习率衰减器
        """

        print('Creating lr scheduler: %s' % lr_scheduler_type)
        if lr_scheduler_type == 'StepLR':
            if not step_size:
                raise ValueError('You must specified step_size when you are using StepLR.')
            my_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
        elif lr_scheduler_type == 'CosineLR':
            if not restart_step:
                raise ValueError('You must specified restart_step when you are using CosineLR.')
            my_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, restart_step)
        elif lr_scheduler_type == 'MultiStepLR':
            if not multi_step:
                raise ValueError('You must specified multi step when you are using MultiStepLR.')
            my_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, multi_step)            
        elif lr_scheduler_type == 'ReduceLR':
            my_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        else:
            raise NotImplementedError("Please supply a right lr_scheduler_tyoe.")

        return my_lr_scheduler