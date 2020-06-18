import argparse
from prepare.prepare import Prepare
from solver import Solver
from utils.parse_config import parse_config
from logger.logger import Logger


class Train(object):
    def __init__(self, config):
        self.train_prepare = Prepare()
        
        self.model = self.train_prepare.create_model(config["model_type"], config["model_cfg"], config["image_size"],
                                                     config["pretrained_weight"], config["freeze_backbone"])
        self.optimizer = self.train_prepare.create_optimizer(self.model, config["optimizer"], config["lr"], config["weight_decay"])
        self.criterion = self.train_prepare.create_criterion(config["loss_type"], config["loss_weights"])
        self.lr_scheduler = self.train_prepare.create_lr_scheduler(
            config["lr_scheduler_type"],
            self.optimizer, 
            config["step_size"],
            config["restart_step"],
            config["multi_step"]
            )
        self.train_dataloader, self.val_dataloader = self.train_prepare.create_dataloader(config)
        self.logger = Logger(config["log_path"], save_weight_interval=config["save_weight_interval"])
        sparsity_train = self.train_prepare.create_sparsity_train(self.model.module.module_defs, config)
        self.solver = Solver(
            self.model, 
            self.criterion, 
            self.optimizer, 
            config, 
            self.train_dataloader,
            self.logger,
            valid_data_loader=self.val_dataloader, 
            lr_scheduler=self.lr_scheduler,
            sparsity_train=sparsity_train
            )

        self.start_epoch = config["start_epoch"]
        self.epoch = config["epoch"]
        self.freeze_backbone = config["freeze_backbone"]
        self.freeze_epoch = config["freeze_epoch"]

    def run(self):
        # TODO 测试冻结backbone
        if self.freeze_backbone:
            for epoch_index in range(self.start_epoch, self.freeze_epoch + 1):
                self.solver.train_epoch(epoch_index)
            print("@Unfreezing backbone.")
            self.model.module.unfreeze_backbone()
            self.optimizer = self.train_prepare.create_optimizer(self.model, config["optimizer"], 
                                                                config["lr_after_freeze"], config["weight_decay"])
            self.lr_scheduler = self.train_prepare.create_lr_scheduler(
                                                                config["lr_scheduler_type"],
                                                                self.optimizer, 
                                                                config["step_size_after_freeze"],
                                                                config["restart_step_after_freeze"],
                                                                config["multi_step_after_freeze"]
                                                            )
            self.solver.update_model(self.model)
            self.solver.update_optimizer(self.optimizer)
            self.solver.update_lr_scheduler(self.lr_scheduler)
            for epoch_index in range(self.freeze_epoch, self.epoch):
                self.solver.train_epoch(epoch_index)
        else:
            for epoch_index in range(self.start_epoch, self.epoch):
                self.solver.train_epoch(epoch_index)


if __name__ == '__main__':
    config = parse_config('config.json')
    train = Train(config)
    train.run()
