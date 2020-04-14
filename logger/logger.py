import os
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_path, save_weight_interval=10):
        self.log_path = log_path
        self.tensorboard_writer, self.weights_path = self.init_log()
        self.save_weight_interval = save_weight_interval

    def save_weight(self, model, epoch):
        if (epoch + 1) % self.save_weight_interval == 0:
            save_path = os.path.join(self.weights_path, "yolov3_" + str(epoch) + ".pth")
            print("@ Saving weight to %s " % save_path)
            torch.save(model.module.state_dict(), save_path)

    def log_list_in_tensorboard(self, lists, epoch):
        for item in lists:
            self.log_in_tensorboard(item[0], item[1], epoch)

    def log_list_in_terminal(self, tag, lists, epoch):
        info = "Epoch: %d - " % epoch + tag
        for item in lists:
            info += item[0] + " %.4f; " % item[1]
        print(info)

    def log_in_tensorboard(self, key, value, step):
        self.tensorboard_writer.add_scalar(key, value, step)

    def log_in_terminal(self, tbar, loss, optimizer=None):
        descript = ""
        descript += "Loss: %.4f " % loss

        if optimizer is not None:
            params_groups_lr = ""
            for group_ind, param_group in enumerate(optimizer.param_groups):
                params_groups_lr = params_groups_lr + 'pg_%d' % group_ind + ': %.8f ' % param_group['lr']
            descript += params_groups_lr

        tbar.set_description(desc=descript)

    def print_in_terminal(self, loss, epoch):
        print("Epoch: %d: %.4f" % (epoch, loss))

    def init_log(self):
        """保存配置信息和初始化tensorboard
        """
        print("Init Logger.")
        TIMESTAMP = "log-{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
        log_dir = os.path.join(self.log_path, TIMESTAMP)
        tensorboard_dir = os.path.join(log_dir, "tensorboard")
        print("@ Tensorboard files will be saved to: %s" % tensorboard_dir)
        tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
        weights_dir = os.path.join(log_dir, "weights")
        os.mkdir(weights_dir)
        print("@ Weights will be saved to: %s" % weights_dir)
        return tensorboard_writer, weights_dir
