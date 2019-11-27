# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-11-26 09:28

"""
第一步
画出人脸边框以及相应关键点以熟悉程序操作、验证坐标表示以及检验标注数据是否正确
"""

from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import cv2

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


for folder in FOLDER_LIST:
    DATA_DIR = ROOT_DIR + folder
    DATA_NAME = os.listdir(DATA_DIR)
    # 整理包含的图片信息
    with open(os.path.join(DATA_DIR, LABEL)) as f:
        contents = f.readlines()
        for content in contents:
            info_list = content.replace("\n", "").split(" ")
            image_name = info_list[0]
            image_rect = [is_bigger_zero(info_list[1]), is_bigger_zero(info_list[2]),
                          is_bigger_zero(info_list[3]), is_bigger_zero(info_list[4])]
            images_landmarks = []
            for i in range(0, len(info_list)-5, 2):
                landmark = (is_bigger_zero(info_list[i+5]), is_bigger_zero(info_list[i+1+5]))
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


def visualize_image():
    """
    随机取一组显示
    :return:
    """
    idx = random.randint(0, len(image_infos))
    image_info = image_infos[idx]
    if image_info['name'] in images:
        image = cv2.imread(image_info['name'], 1)
        # 画人脸矩形框
        rect = image_info['rect']
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0))
        # 画关键点
        landmarks = image_info['landmarks']
        for c in range(0, len(landmarks)):
            center = landmarks[c]
            cv2.circle(image, center, 2, (0, 0, 255), -1)
        image_new = image_bgr_to_rgb(image)
        plt.imshow(image_new)
        plt.show()


# 随机取一组显示
visualize_image()
