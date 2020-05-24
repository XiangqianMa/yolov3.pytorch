import torch
from compression.prune.parse_prune_cfg import parse_prune_cfg


class SparsityTrain:
    def __init__(self, module_defs, s):
        self.s = s
        self.prune_idx = parse_prune_cfg(module_defs)[-1]

    def update_bn(self, module_list):
        for idx in self.prune_idx:
            bn_module = module_list[idx][1]
            bn_module.weight.grad.data.add_(self.s * torch.sign(bn_module.weight.data))
