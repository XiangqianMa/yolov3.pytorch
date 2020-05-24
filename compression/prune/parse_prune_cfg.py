import torch
import numpy as np
import torch.nn.functional as F


def parse_prune_cfg(module_defs):
    """
    解析出需要进行剪枝操作的层的索引
    :param module_defs: 模型结构定义
    :return: conv_bn_idx: conv2d + bn2d + leaky ReLu 的索引
             conv_idx: conv2d的索引
             prune_idx: 待剪枝的层的索引
    """
    conv_bn_idx = []
    conv_idx = []
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                conv_bn_idx.append(i)
            else:
                conv_idx.append(i)

    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'shortcut':
            ignore_idx.add(i - 1)
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                ignore_idx.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                ignore_idx.add(identity_idx - 1)
    # 两个上采样层
    ignore_idx.add(84)
    ignore_idx.add(96)

    prune_idx = [idx for idx in conv_bn_idx if idx not in ignore_idx]

    return conv_bn_idx, conv_idx, prune_idx
