import os

import numpy as np
import cv2
from random import randint
from matplotlib import pyplot as plt


def random_colors(class_number):
    colors = []
    for i in range(class_number):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        colors.append((r, g, b))
    return colors


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=(255, 0, 0), text_color=(255, 255, 255), thickness=2):
    height, width = img.shape[0], img.shape[1]
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    x_min = 0 if x_min <= 0 else x_min
    y_min = 0 if y_min <= 0 else y_min
    x_max = width if x_max >= width else x_max
    y_max = height if y_max >= height else y_max
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[str(class_id)]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color,
                lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name, show=True):
    colors = random_colors(len(category_id_to_name))
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        class_id = annotations['category_id'][idx]
        color = colors[class_id]
        img = visualize_bbox(img, bbox, class_id, category_id_to_name, color=color)
    if show:
        plt.figure(figsize=(12, 12))
        plt.imshow(img)
        plt.show()
    return img
