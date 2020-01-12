import argparse
from prepare.prepare import Prepare
from trainer import TrainVal


class Train():
    def __init__(self, config):
        train_prepare = Prepare()
        
        self.model = train_prepare.create_model(config.model_type, config.model_cfg, config.pretrained_weight)
        self.optimizer = train_prepare.create_optimizer(self.model, config)
        self.criterion = train_prepare.create_criterion(config.loss_type, config.loss_weights)
        self.lr_scheduler = train_prepare.create_lr_scheduler(
            config.lr_scheduler_type, 
            self.optimizer, 
            config.step_size, 
            config.restart_step, 
            config.multi_step
            )
        self.trian_dataloader, self.val_dataloader = train_prepare.create_dataloader(config)
        self.train_val = TrainVal(
            self.model, 
            self.criterion, 
            self.optimizer, 
            config, 
            self.trian_dataloader, 
            valid_data_loader=self.val_dataloader, 
            lr_scheduler=self.lr_scheduler
            )
        
        self.epoch = config.epoch

    def run(self):
        for epoch_index in len(self.epoch):
            self.train_val._train_epoch(epoch_index)


if __name__ == '__main__':

    train = Train(config)
    train.run()