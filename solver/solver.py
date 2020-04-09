import numpy as np
import torch
import tqdm
from torchvision.utils import make_grid
from utils import inf_loop, MetricTracker


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
        for _, data, target in tbar:
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.logger.log_in_terminal(tbar, loss.item(), self.optimizer)
            self.logger.save_weight(self.model, epoch)

        if self.do_validation:
            self._valid_epoch(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _valid_epoch(self, epoch):
        """
        完成一个epoch的训练以后使用验证集进行验证

        Args:
            epoch: 当前epoch的index
        """
        self.model.eval()
        tbar = tqdm.tqdm(self.valid_data_loader)
        with torch.no_grad():
            for _, data, target in tbar:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)  # TODO: 当前损失无法被用于验证

        # add histogram of models parameters to the tensorboard
