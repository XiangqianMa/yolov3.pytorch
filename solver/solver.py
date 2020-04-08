import numpy as np
import torch
from torchvision.utils import make_grid
from utils import inf_loop, MetricTracker


class Solver:
    """
    Train and val class
    """
    def __init__(self, model, criterion, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None):
        self.config = config
        self.data_loader = data_loader

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.lr_scheduler = lr_scheduler

    def train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()

        for _, data, target in self.data_loader:
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(loss.item())

        if self.do_validation:
            self._valid_epoch(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

        # add histogram of models parameters to the tensorboard

