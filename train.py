import argparse
from prepare.prepare import Prepare
from solver import Solver
from utils.parse_config import parse_config
from logger.logger import Logger


class Train(object):
    def __init__(self, config):
        train_prepare = Prepare()
        
        self.model = train_prepare.create_model(config["model_type"], config["model_cfg"], config["image_size"], config["pretrained_weight"])
        self.optimizer = train_prepare.create_optimizer(self.model, config)
        self.criterion = train_prepare.create_criterion(config["loss_type"], config["loss_weights"])
        self.lr_scheduler = train_prepare.create_lr_scheduler(
            config["lr_scheduler_type"],
            self.optimizer, 
            config["step_size"],
            config["restart_step"],
            config["multi_step"]
            )
        self.train_dataloader, self.val_dataloader = train_prepare.create_dataloader(config)
        self.logger = Logger(config["log_path"], save_weight_interval=config["save_weight_interval"])
        self.solver = Solver(
            self.model, 
            self.criterion, 
            self.optimizer, 
            config, 
            self.train_dataloader,
            self.logger,
            valid_data_loader=self.val_dataloader, 
            lr_scheduler=self.lr_scheduler
            )

        self.start_epoch = config["start_epoch"]
        self.epoch = config["epoch"]

    def run(self):
        for epoch_index in range(self.start_epoch, self.epoch):
            self.solver.train_epoch(epoch_index)


if __name__ == '__main__':
    config = parse_config('config.json')
    train = Train(config)
    train.run()
