import os
import numpy as np


def parse_model_cfg(path):
    """解析yolo*.cfg中解析出yolo的网络结构配置信息

    Args:
        path: .cfg文件的路径
    Return:
        module_definitions: list, 网络中各个模块的配置信息
    """
    # 解析得到cfg文件的正确路径
    if not path.endswith('.cfg'):
        path += '.cfg'
    if not os.path.exists(path):
        raise ValueError('No %s' % path)

    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_definitions = []  # module definitions
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_definitions.append({})
            module_definitions[-1]['type'] = line[1:-1].rstrip()
            if module_definitions[-1]['type'] == 'convolutional':
                module_definitions[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=")
            key = key.rstrip()

            # 解析anchor的大小
            if 'anchors' in key:
                module_definitions[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
            else:
                module_definitions[-1][key] = val.strip()

    # 检查所支持的module类型
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y']

    f = []  # fields
    for x in module_definitions[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]  # unsupported fields
    assert not any(u), "Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631" % (u, path)

    return module_definitions


def parse_data_cfg(path):
    """解析data配置文件

    Args:
        path: .data文件的路径, *.data
    Return:
        options: list, 数据的配置信息
    """
    # Parses the data configuration file
    if not os.path.exists(path):  # add data/ prefix if omitted
        raise ValueError('No %s' % path)

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options


if __name__ == '__main__':
    model_cfg_path = 'cfg/model_cfg/yolov3.cfg'
    module_definitions = parse_model_cfg(model_cfg_path)
    data_cfg_path = 'cfg/data_cfg/coco1.data'
    data_cfg = parse_data_cfg(data_cfg_path)
    pass
