# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-11-26 19:12

"""
生成训练集和测试集
"""

import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from random import shuffle

ROOT_DIR = '../Data/'
FOLDER_LIST = ['I', 'II']
LABEL = 'label.txt'

images = []
image_infos = []


def is_bigger_zero(a):
    """
    将图像坐标转换成整数
    :param a:
    :return:
    """
    if int(float(a)) < 0:
        return 0
    else:
        return int(float(a))


def image_bgr_to_rgb(old_img):
    """
    将BGR表示的图像转换成RGB表示的图像
    用于OpenCV与PIL使用的图像格式之间的转换
    :param old_img:
    :return:
    """
    (b, g, r) = cv2.split(old_img)
    img_new = cv2.merge((r, g, b))
    return img_new


def expand_roi(x1, y1, x2, y2, img_width, img_height, ratio=0.25):
    """
    扩增矩形框
    :param x1: 原矩形框左上角顶点x坐标
    :param y1: 原矩形框左上角顶点y坐标
    :param x2: 原矩形框右下角顶点x坐标
    :param y2: 原矩形框右下角顶点y坐标
    :param img_width: 图像宽度
    :param img_height: 图像高度
    :param ratio: 扩增倍数，默认0.25倍
    :return:
    """
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    padding_width = int(width * ratio)
    padding_height = int(height * ratio)
    roi_x1 = x1 - padding_width
    roi_y1 = y1 - padding_height
    roi_x2 = x2 + padding_width
    roi_y2 = y2 + padding_height
    # 边界判断 扩增后的人脸框不要超过原图像大小
    roi_x1 = 0 if roi_x1 < 0 else roi_x1
    roi_y1 = 0 if roi_y1 < 0 else roi_y1
    roi_x2 = img_width - 1 if roi_x2 >= img_width else roi_x2
    roi_y2 = img_height - 1 if roi_y2 >= img_height else roi_y2
    return roi_x1, roi_y1, roi_x2, roi_y2, roi_x2 - roi_x1 + 1, roi_y2 - roi_y1 + 1


for folder in FOLDER_LIST:
    DATA_DIR = ROOT_DIR + folder
    DATA_NAME = os.listdir(DATA_DIR)
    # 整理包含的图片信息
    with open(os.path.join(DATA_DIR, LABEL)) as f:
        contents = f.readlines()
        for content in contents:
            info_list = content.replace("\n", "").split(" ")
            image_name = info_list[0]
            # 扩增矩形框
            expend_img = cv2.imread(os.path.join(DATA_DIR, image_name), 1)
            h, w, channel = expend_img.shape
            expend_x1, expend_y1, expend_x2, expend_y2, new_width, new_height = expand_roi(
                is_bigger_zero(info_list[1]), is_bigger_zero(info_list[2]),
                is_bigger_zero(info_list[3]), is_bigger_zero(info_list[4]),
                w, h, ratio=0.25
            )
            image_rect = [expend_x1, expend_y1, expend_x2, expend_y2]
            images_landmarks = []
            for i in range(0, len(info_list) - 5, 2):
                landmark = (is_bigger_zero(info_list[i + 5]), is_bigger_zero(info_list[i + 1 + 5]))
                images_landmarks.append(landmark)
            image_infos.append({"name": os.path.join(DATA_DIR, image_name),
                                "rect": image_rect,
                                "landmarks": images_landmarks})
    # 整理图片
    for item in DATA_NAME:
        try:
            img = Image.open(os.path.join(DATA_DIR, item))
        except OSError:
            pass
        else:
            images.append(os.path.join(DATA_DIR, item))

train_test_infos = []

# 截取人脸，并将人脸关键点坐标变为相对于截取后的人脸图的坐标
for info in image_infos:
    if info['name'] in images:
        image = cv2.imread(info['name'], 1)
        train_test_str = info['name']
        # 人脸矩形框
        rect = info['rect']
        for rect_coor in rect:
            train_test_str += " " + str(rect_coor)
        # 关键点
        landmarks = info['landmarks']
        for c in range(0, len(landmarks)):
            center = landmarks[c]
            center -= np.array([rect[0], rect[1]])
            for center_coor in center:
                train_test_str += " " + str(center_coor)
        train_test_infos.append(train_test_str)

# 验证正确性
idx = random.randint(0, len(train_test_infos))
train_test_val = train_test_infos[idx]
train_test = train_test_val.split(" ")
image = cv2.imread(train_test[0], 1)
print(train_test[0])
# 画人脸矩形框
cv2.rectangle(image,
              (int(train_test[1]), int(train_test[2])), (int(train_test[3]), int(train_test[4])),
              (0, 255, 0), thickness=2)
# 画关键点
for i in range(0, len(train_test)-5, 2):
    # 由于关键点坐标是相对于人脸矩形框的，绘制时需要调整
    center = (int(train_test[i+5])+int(train_test[1]), int(train_test[i+1+5])+int(train_test[2]))
    cv2.circle(image, center, 2, (0, 0, 255), -1)
image_new = image_bgr_to_rgb(image)
plt.imshow(image_new)
plt.show()

# 将数据分成训练集和测试集 80%训练集 20%测试集
shuffle(train_test_infos)
split_idx = int(len(train_test_infos) * 0.8)
with open("train.txt", "a+") as f:
    for i in range(split_idx):
        train_info = train_test_infos[i]
        f.write(train_info + "\n")
with open("test.txt", "a+") as f:
    for i in range(split_idx, len(train_test_infos)):
        test_info = train_test_infos[i]
        f.write(test_info + "\n")
