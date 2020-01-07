import cv2
import numpy as np

image_path = "/media/mxq/project/Projects/object_detection/yolov3.pytorch/data/coco/val2017/000000000139.jpg"

x = 0.686
y = 0.5319
w = 0.0828
h = 0.323

img = cv2.imread(image_path)
height = np.shape(img)[0]
weight = np.shape(img)[1]

print(weight, height)

print(x*weight, y*height, w*weight, h*height)

left = int(x * weight - w * (weight/2))
right = int(x * weight + w * (weight/2))
top = int(y * height - h * (height/2))
bottom = int(y * height + h * (height/2))

cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
cv2.imshow("img1", img)
cv2.waitKey(0)