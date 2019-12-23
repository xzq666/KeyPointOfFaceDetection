# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-11-26 18:04

import numpy as np

arr = np.random.rand(112, 112, 1)
print("----------------")
img = arr.transpose((2, 0, 1))
img = np.expand_dims(img, axis=0)
img_test = np.squeeze(img, axis=(1,))
img_test = img_test.transpose((1, 2, 0))
print(img_test == arr)
