# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-04 15:49

import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import random

train_boarder = 112


def channel_norm(img):
    """
    将图像进行通道归一化
    :param img: ndarray, float32
    :return:
    """
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels


def parse_line(line):
    """
    解析从txt文件中读取的每一行
    :param line:
    :return:
    """
    line_parts = line.strip().split()
    img_name = line_parts[0]
    rect = list(map(int, list(map(float, line_parts[1:5]))))
    landmarks = list(map(float, line_parts[5:len(line_parts)]))
    return img_name, rect, landmarks


class Normalize(object):
    """
    重新缩放尺寸并进行通道归一化(image - mean) / std_variation
    """
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image_resize = np.asarray(image.resize((train_boarder, train_boarder), Image.BILINEAR), dtype=np.float32)
        image = channel_norm(image_resize)
        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """
    将ndarrays转换为张量Tensor
    张量通道序列: N x C x H x W
    """
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        # image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=2)
        print(image.shape)
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}


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
        img_name, rect, landmarks = parse_line(self.lines[idx])
        # image
        img = Image.open(img_name).convert('L')
        img_crop = img.crop(tuple(rect))
        landmarks = np.array(landmarks).astype(np.float32)

        # you should let your landmarks fit to the train_boarder(112)
        # please complete your code under this blank
        # your code:

        sample = {'image': img_crop, 'landmarks': landmarks}
        sample = self.transform(sample)
        return sample


def load_data(phase):
    data_file = phase + '.txt'
    with open(data_file) as f:
        lines = f.readlines()
    if phase == 'Train' or phase == 'train':
        tsfm = transforms.Compose([
            Normalize(),                # do channel normalization
            ToTensor()]                 # convert to torch type: NxCxHxW
        )
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    data_set = FaceLandmarksDataset(lines, phase, transform=tsfm)
    return data_set


def get_train_test_set():
    train_set = load_data('train')
    valid_set = load_data('test')
    return train_set, valid_set


if __name__ == '__main__':
    train_set = load_data('train')
    sample = train_set[0]
    img = sample['image']
    landmarks = sample['landmarks']
    # 请画出人脸crop以及对应的landmarks
    # please complete your code under this blank
    cv2.imshow("image", img.numpy())
    cv2.waitKey(0)
