import xml.etree.ElementTree as ET
import os
import shutil
import tqdm
import json
from os import getcwd


sets = ['train', 'test']
# classes
classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
# the path of JPEGImages
image_path = "data/voc/JPEGImages"
# the path where you want to put your label files in
target_label_path_root = "data/voc"
# the path of original annotation
annotation_path = "data/voc/Annotations"
# the path where you want to put your image path files in
target_image_path_root = "data/voc"
# the path of train.txt and val.txt
dataset_separate_path = "data/voc/ImageSets/Main"
json_path = "data/voc/categories_id_to_name.json"


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x, y, w, h


def convert_annotation(image_id, target_label_path):
    in_file_name = os.path.join(annotation_path, image_id + '.xml')
    out_file_name = os.path.join(target_label_path, image_id + '.txt')
    in_file = open(in_file_name)
    out_file = open(out_file_name, 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# the current working dirt
wd = getcwd()

for image_set in sets:
    target_image_path = os.path.join(target_image_path_root, image_set)
    target_label_path = os.path.join(target_label_path_root, image_set + "_txt")
    if not os.path.exists(target_image_path):
        os.makedirs(target_image_path)
    if not os.path.exists(target_label_path):
        os.makedirs(target_label_path)

    image_set_txt = os.path.join(dataset_separate_path, image_set + '.txt')
    image_ids = open(image_set_txt).read().strip().split()

    tbar = tqdm.tqdm(image_ids)
    for image_id in tbar:
        image_path_0 = os.path.join(image_path, image_id + '.jpg')
        image_path_1 = os.path.join(target_image_path, image_id + '.jpg')
        descript = "Copy from %s to %s" % (image_path_0, image_path_1)
        shutil.copy(image_path_0, image_path_1)
        convert_annotation(image_id, target_label_path)
        tbar.set_description(desc=descript)

id_to_name = {}
for index, class_name in enumerate(classes):
    id_to_name[str(index)] = class_name
with open(json_path, 'w') as f:
    json.dump(id_to_name, f)
