import numpy as np
import torch
import tqdm
from torchvision.utils import make_grid
from utils.evaluate import evaluate


class Solver:
    """
    完成训练、验证、中间结果可视化、模型保存等操作。
    """
    def __init__(self, model, criterion, optimizer, config, data_loader, logger,
                 valid_data_loader=None, lr_scheduler=None):
        self.config = config
        self.data_loader = data_loader

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.logger = logger

        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.lr_scheduler = lr_scheduler

    def train_epoch(self, epoch):
        """
        完成一个epoch的训练

        Args:
            epoch: 当前epoch的索引
        """
        self.model.train()

        tbar = tqdm.tqdm(self.data_loader)
        epoch_loss = 0
        for iter_index, (_, data, target) in enumerate(tbar):
            data, target = data.cuda(), target.cuda()

            output = self.model(data)
            loss = self.criterion(output, target)
            self._update_parameter(loss, iter_index)

            epoch_loss += loss.item()
            self.logger.log_in_terminal(tbar, loss.item(), self.optimizer)
            self.logger.log_in_tensorboard("iter-loss", loss.item(), epoch * len(tbar) + iter_index)

        self.logger.print_in_terminal(epoch_loss / len(tbar), epoch)
        self.logger.log_in_tensorboard("epoch-loss", epoch_loss / len(tbar), epoch)
        self.logger.save_weight(self.model, epoch)

        if self.do_validation and (epoch + 1) % self.config["val_interval"] == 0:
            self._valid_epoch(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _valid_epoch(self, epoch):
        """
        完成一个epoch的训练以后使用验证集进行验证

        Args:
            epoch: 当前epoch的index
        """
        precision, recall, AP, f1, ap_class = evaluate(
            self.model,
            self.valid_data_loader,
            self.config["iou_thres"],
            self.config["conf_thres"],
            self.config["nms_thres"],
            self.config["image_size"][0]
        )

        evaluation_metrics = [
            ("val_precision", precision.mean()),
            ("val_recall", recall.mean()),
            ("val_mAP", AP.mean()),
            ("val_f1", f1.mean()),
        ]

        self.logger.log_list_in_terminal("Val: ", evaluation_metrics, epoch)
        self.logger.log_list_in_tensorboard(evaluation_metrics, epoch)

    def _update_parameter(self, loss, iterations):
        loss.backward()
        if self.config["gradient_accumulation"] is not None:
            if iterations % self.config["gradient_accumulation"] == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
