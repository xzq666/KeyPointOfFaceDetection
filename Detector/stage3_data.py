# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-25 15:46

from torchvision import transforms
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2
import random

# train_boarder = 112
train_boarder = 224


def parse_line(line):
    """
    解析从txt文件中读取的每一行
    :param line:
    :return:
    """
    line_parts = line.strip().split()
    img_name = line_parts[0]
    rect = list(map(int, list(map(float, line_parts[1:5]))))
    face = line_parts[-1]
    landmarks = list(map(float, line_parts[5:len(line_parts) - 1])) if (int(face) == 1) \
        else [0.0 * c for c in range(42)]
    return img_name, rect, landmarks, face


class FaceLandmarksDataset(Dataset):
    """
    自定义数据集
    """

    def __init__(self, src_lines, phase, transform=None):
        """
        :param src_lines: src_lines
        :param phase: whether we are training or not
        :param transform: data transform
        """
        self.lines = src_lines
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_name, rect, landmarks, face = parse_line(self.lines[idx])
        # image
        img = Image.open(img_name).convert('L')
        img_crop = img.crop(tuple(rect))
        landmarks = np.array(landmarks).astype(np.float32)

        if int(face) == 1:
            origin_width = rect[2] - rect[0]
            origin_height = rect[3] - rect[1]
            w_ratios = train_boarder / origin_width
            h_ratios = train_boarder / origin_height
            for k in range(0, len(landmarks), 2):
                landmarks[k] = round(landmarks[k] * w_ratios)
                landmarks[k + 1] = round(landmarks[k + 1] * h_ratios)

        sample = {'image': img_crop, 'landmarks': landmarks, 'face': face}
        sample = self.transform(sample)
        return sample


class ToTensor(object):
    """
    将ndarrays转换为张量Tensor
    张量通道序列: N x C x H x W
    """

    def __call__(self, sample):
        image, landmarks, face = sample['image'], sample['landmarks'], sample['face']
        # face_numpy = np.array(list([float(face)]))
        image = np.expand_dims(image, axis=2)
        image = cv2.resize(image, (train_boarder, train_boarder))
        image = np.expand_dims(image, axis=2)
        # 使用ResNet18时需要将图像转为彩色图
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks),
                'face': int(face)}


def load_data(phase):
    """
    去除normalize
    :param phase:
    :return:
    """
    data_file = phase + '.txt'
    with open(data_file) as f:
        lines = f.readlines()
    if phase == 'stage2_train':
        tsfm = transforms.Compose([
            ToTensor()
        ])
    else:
        tsfm = transforms.Compose([
            ToTensor()
        ])
    data_set = FaceLandmarksDataset(lines, phase, transform=tsfm)
    return data_set


def get_train_test_set():
    train_set = load_data('stage3_train')
    valid_set = load_data('stage3_test')
    return train_set, valid_set


if __name__ == '__main__':
    train_sets = load_data('stage3_train')
    idx_test = random.randint(0, len(train_sets))
    sample_test = train_sets[idx_test]
    print(sample_test)
    img_test = sample_test['image']
    # 将Tensor格式转换成OpenCV的图像格式
    img_test = img_test.numpy()
    # img_test = np.squeeze(img_test, axis=(1,))
    img_test = img_test.transpose((1, 2, 0))
    # 调用下面的cv2.circle时
    # 由于这里对img_test有数据操作，当传入circle函img_copy数是不连续的内存数据，
    # 而该函数输出的内存是连续的
    # 为了保证输入输出一致，这里调用copy()方法获取连续的内存数据img_copy
    img_copy = img_test.copy()
    landmarks_test = sample_test['landmarks']
    face_test = sample_test['face']
    if face_test == 1:
        # 请画出人脸crop以及对应的landmarks
        # please complete your code under this blank
        for i in range(0, len(landmarks_test), 2):
            # 由于关键点坐标是相对于人脸矩形框的，绘制时需要调整
            center = (int(landmarks_test[i]), int(landmarks_test[i + 1]))
            cv2.circle(img_copy, center, 1, (255, 0, 0), -1)
    cv2.imshow("image", img_copy)
    cv2.waitKey(0)
