import xml.etree.ElementTree as ET
import os
import shutil
import tqdm
import json
from os import getcwd


sets = ['test']
# classes
classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
# the path of JPEGImages
image_path = "data/VOCdevkit/JPEGImages"
# the path where you want to put your label files in
target_label_path = "data/voc/groundtruths"
# the path of original annotation
annotation_path = "data/VOCdevkit/Annotations"
# the path of train.txt and val.txt
dataset_separate_path = "data/VOCdevkit/ImageSets/Main"


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
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymax').text))
        out_file.write(cls + " " + " ".join([str(a) for a in b]) + '\n')


# the current working dirt
wd = getcwd()

for image_set in sets:
    if not os.path.exists(target_label_path):
        os.makedirs(target_label_path)

    image_set_txt = os.path.join(dataset_separate_path, image_set + '.txt')
    image_ids = open(image_set_txt).read().strip().split()

    tbar = tqdm.tqdm(image_ids)
    for image_id in tbar:
        print(image_id)
        convert_annotation(image_id, target_label_path)

