# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-12-24 14:57

import os
from PIL import Image
import cv2
import random
import numpy as np

ROOT_DIR = '../Data/'
FOLDER_LIST = ['I', 'II']
SAVE_LIST = 'stage2/'

SAVE_DIR = ROOT_DIR + SAVE_LIST

for folder in FOLDER_LIST:
    DATA_DIR = ROOT_DIR + folder
    DATA_NAME = os.listdir(DATA_DIR)
    # 整理图片
    for item in DATA_NAME:
        try:
            img = Image.open(os.path.join(DATA_DIR, item))
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        except OSError:
            print(item)
            pass
        except Exception:
            print(item)
            pass
        else:
            # 对图像进行亮度与饱和度改变做图像增广
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            img_copy = img.copy()
            # 饱和度
            img_copy[:, :, 2] = (1.0 + random.randint(1, 100) / 100.0) * img_copy[:, :, 2]
            img_copy[:, :, 2][img_copy[:, :, 2] > 1] = 1
            lsImg = cv2.cvtColor(img_copy, cv2.COLOR_HLS2BGR)
            cv2.imwrite(SAVE_DIR + item, lsImg)
