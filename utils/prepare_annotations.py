#
# 将各数据集的原始标注转换为yolo所需的形式
#
import os
import json
from tqdm import tqdm
from utils.bbox_convert import upleft_to_center


def prepare_coco_annotations(annotation_file):
    """从coco的原始标定文件中抽取目标实例的标注信息

    Args:
        annotation_file: 原始json形式的标注文件
    Return:
        images_to_annotations: dir, 文件到其标定的映射， {'000000289343.jpg': [[category_id, [x, y, width, height]], ...], ...}
        images_to_width_height: dir, 文件到其宽和高的映射， {'000000289343.jpg': {'width': 416, 'height': 416}, ...}
        categories_id_to_name: dir, 类别id到类别名称的映射, {1: 'person', ...}
    """
    with open(annotation_file, 'r') as f:
        annotation = json.load(f)

    annotations = annotation['annotations']
    images = annotation['images']
    categories = annotation['categories']

    categories_id_to_name = {}
    for category in categories:
        categories_id_to_name[category['id']] = category['name']

    # 图片信息，包含id到filename的对应关系，图片的width, height
    images_info = {}
    for image in images:
        image_id = image['id']
        file_name = image['file_name']
        height, width = image['height'], image['width']
        images_info[image_id] = {'file_name': file_name, 'height': height, 'width': width}

    # 图片名称到其宽和高的映射
    images_to_width_height = {}
    for value in images_info.values():
        images_to_width_height[value['file_name']] = {'width': value['width'], 'height': value['height']}

    images_to_annotations = {}
    for annotation_instance in annotations:
        image_id = annotation_instance['image_id']
        category_id = annotation_instance['category_id']
        bbox = annotation_instance['bbox']
        file_name = images_info[image_id]['file_name']
        instance = [category_id, bbox]
        if file_name not in images_to_annotations.keys():
            images_to_annotations[file_name] = [instance]
        else:
            images_to_annotations[file_name].append(instance)

    return images_to_annotations, images_to_width_height, categories_id_to_name


def annotations_to_txt(
    images_to_annotations, 
    images_to_width_height, 
    categories_id_to_name, 
    txt_folder, 
    categories_name_to_id_json, 
    categories_id_to_name_json
    ):

    """将各个图片文件对应的标注信息转换为yolo的格式，并写入txt文件中

    Args:
        images_to_annotations: dir, 文件到其标定的映射， {'000000289343.jpg': [[category_id, [x, y, width, height]], ...], ...}
        images_to_width_height: dir, 文件到其宽和高的映射， {'000000289343.jpg': {'width': 416, 'height': 416}, ...}
        categories_id_to_name: dir, 类别id到类别名称的映射, {1: 'person', ...}
        txt_folder: 保存txt文件的文件夹的路径
        categories_name_to_id_json: json文件的路径，保存类别名称到id的映射
    """
    categories_name = []
    for category_name in categories_id_to_name.values():
        categories_name.append(category_name)
    # 类别名到0~len(categories)的转换，coco的官方id不是连续的
    categories_name_to_id = {category_name: index for index, category_name in enumerate(categories_name)}

    tbar = tqdm(images_to_annotations.items())
    for image_file_name, bboxes in tbar:
        image_width, image_height = images_to_width_height[image_file_name]['width'], images_to_width_height[image_file_name]['height']
        with open(os.path.join(txt_folder, image_file_name.replace('jpg', 'txt')), 'w') as annotation_txt:
            for bbox in bboxes:
                category_name = categories_id_to_name[bbox[0]]
                bbox_xywh = bbox[1]
                center_x_ratio, center_y_ratio, width_ratio, height_ratio = upleft_to_center(
                    bbox_xywh[0],
                    bbox_xywh[1],
                    bbox_xywh[2],
                    bbox_xywh[3],
                    image_width,
                    image_height)
                category_id = categories_name_to_id[category_name]
                annotation_line = \
                    str(category_id) + ' ' \
                    + str(center_x_ratio) + ' ' \
                    + str(center_y_ratio) + ' ' \
                    + str(width_ratio) + ' ' \
                    + str(height_ratio) + '\n'
                annotation_txt.writelines(annotation_line)
        tbar.set_description(desc=image_file_name)
    with open(categories_name_to_id_json, 'w') as f:
        print('@ Writing categories_name_to_id to %s' % categories_name_to_id_json)
        json.dump(categories_name_to_id, f)
    categories_id_to_name = {}
    for key, value in categories_name_to_id.items():
        categories_id_to_name[int(value)] = key
    with open(categories_id_to_name_json, 'w') as f:
        print('@ Writing categories_id_to_name_json to %s' % categories_id_to_name_json)
        json.dump(categories_id_to_name, f)


if __name__ == '__main__':
    train_annotation = 'data/coco/annotations_trainval2017/instances_val2017.json'
    annotation_txt_folder = 'data/coco/val2017_txt'
    categories_name_to_id_json = 'data/coco/categories_name_to_id.json'
    categories_id_to_name_json = 'data/coco/categories_id_to_name.json'
    images_to_annotations, images_to_width_height, categories_id_to_name = prepare_coco_annotations(train_annotation)
    annotations_to_txt(images_to_annotations, images_to_width_height, categories_id_to_name, annotation_txt_folder, categories_name_to_id_json, categories_id_to_name_json)
    pass
