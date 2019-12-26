# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-24 19:03

import os
import cv2
import numpy as np
from PIL import Image
from random import shuffle

ROOT_DIR = '../Data/'
FOLDER_LIST = ['I', 'II']
SAVE_LIST = 'stage3/'
LABEL = 'label.txt'


def is_bigger_zero(a):
    """
    将图像坐标转换成整数
    :param a:
    :return:
    """
    if float(a) < 0:
        return 0
    else:
        return float(a)


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


def random_crop(image, crop_width, crop_height):
    """
    随机裁剪
    :param image:
    :param crop_width:
    :param crop_height:
    :return:
    """
    height, width = image.shape[:2]
    top = np.random.randint(0, height - crop_height)
    left = np.random.randint(0, width - crop_width)
    right = int(left + crop_width)
    bottom = int(top + crop_height)
    return [top, left, bottom, right]


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # 计算每个s矩形的面积
    rec1_s = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    rec2_s = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    # 总面积
    sum_area = rec1_s + rec2_s
    # 获取相交矩形的每个边
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
    # 判断是否有相交
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


train_test_infos = []
images = []
num = 0
for folder in FOLDER_LIST:
    DATA_DIR = ROOT_DIR + folder
    DATA_NAME = os.listdir(DATA_DIR)
    # 整理图片
    for item in DATA_NAME:
        try:
            img = Image.open(os.path.join(DATA_DIR, item))
        except OSError:
            pass
        else:
            images.append(os.path.join(DATA_DIR, item))
    # 整理包含的图片信息
    with open(os.path.join(DATA_DIR, LABEL)) as f:
        contents = f.readlines()
        for content in contents:
            info_list = content.replace("\n", "").split(" ")
            image_name = info_list[0]
            if os.path.join(DATA_DIR, image_name) in images:
                # 扩增矩形框
                expend_img = cv2.imread(os.path.join(DATA_DIR, image_name), 1)
                h, w, channel = expend_img.shape
                expend_x1, expend_y1, expend_x2, expend_y2, new_width, new_height = expand_roi(
                    is_bigger_zero(info_list[1]), is_bigger_zero(info_list[2]),
                    is_bigger_zero(info_list[3]), is_bigger_zero(info_list[4]),
                    w, h, ratio=0.25
                )
                image_rect = [expend_x1, expend_y1, expend_x2, expend_y2]
                # 生成正样本（人脸）数据
                train_test_str_face = os.path.join(DATA_DIR, image_name)
                for rect_coor in image_rect:
                    train_test_str_face += " " + str(rect_coor)
                # 关键点
                landmarks = []
                for i in range(0, len(info_list) - 5, 2):
                    landmark = (is_bigger_zero(info_list[i + 5]), is_bigger_zero(info_list[i + 1 + 5]))
                    landmarks.append(landmark)
                for c in range(0, len(landmarks)):
                    center = landmarks[c]
                    center -= np.array([image_rect[0], image_rect[1]])
                    for center_coor in center:
                        train_test_str_face += " " + str(center_coor)
                train_test_str_face += " 1"
                train_test_infos.append(train_test_str_face)
                # 生成负样本（非人脸）数据
                # 为保证正负样本比例为1:1，直到截到负样本为止
                times = 0
                print("开始生成负样本")
                while True:
                    if times == 100:
                        # 超过100次放弃
                        print("放弃")
                        break
                    crop_rect = random_crop(expend_img, expend_x2 - expend_x1, expend_y2 - expend_y1)
                    iou = compute_iou(image_rect, crop_rect)
                    if iou < 0.3:
                        num += 1
                        print("生成第{}个负样本".format(num))
                        train_test_str = os.path.join(DATA_DIR, image_name)
                        for rect_coor in crop_rect:
                            train_test_str += " " + str(float(rect_coor))
                        train_test_str += " 0"
                        train_test_infos.append(train_test_str)
                        break
                    times += 1

# 将数据分成训练集和测试集 80%训练集 20%测试集
shuffle(train_test_infos)
split_idx = int(len(train_test_infos) * 0.8)
with open("stage3_train.txt", "a+") as f:
    for i in range(split_idx):
        train_info = train_test_infos[i]
        f.write(train_info + "\n")
with open("stage3_test.txt", "a+") as f:
    for i in range(split_idx, len(train_test_infos)):
        test_info = train_test_infos[i]
        f.write(test_info + "\n")
