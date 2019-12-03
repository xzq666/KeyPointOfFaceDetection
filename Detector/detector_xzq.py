# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2019-11-27 15:27

import torch
import torch.nn as nn
import torch.optim as optim
import argparse


class Net(nn.Module):
    """
    根据detector_stage1.prototxt构建网络
    """
    def __init__(self):
        super(Net, self).__init__()
        # input_channel output_channel kernel_size stride padding
        self.conv1_1 = nn.Conv2d(1, 8, 5, 2, 0)
        self.conv2_1 = nn.Conv2d(8, 16, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(16, 16, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(16, 24, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(24, 24, 3, 1, 0)
        self.conv4_1 = nn.Conv2d(24, 40, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(40, 80, 3, 1, 1)
        self.ip1 = nn.Linear(4 * 4 * 80, 128)
        self.ip2 = nn.Linear(128, 128)
        self.landmarks = nn.Linear(128, 42)
        # 需要的激活函数
        self.prelu1_1 = nn.PReLU()
        self.prelu2_1 = nn.PReLU()
        self.prelu2_2 = nn.PReLU()
        self.prelu3_1 = nn.PReLU()
        self.prelu3_2 = nn.PReLU()
        self.prelu4_1 = nn.PReLU()
        self.prelu4_2 = nn.PReLU()
        self.prelu_ip1 = nn.PReLU()
        self.prelu_ip2 = nn.PReLU()
        # 需要的池化层 kernel_size stride ceil_mode:是否舍弃多余边
        self.avg_pool = nn.AvgPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        """
        网络前向传播
        :param x: shape为1x1x112x112
        :return:
        """
        # conv1_1 112->54
        x = self.conv1_1(x)
        # relu_conv1_1
        x = self.prelu1_1(x)
        # pool1 54->27
        x = self.avg_pool(x)
        # conv2_1 27->25
        x = self.conv2_1(x)
        # relu_conv2_1
        x = self.prelu2_1(x)
        # conv2_2 25->23
        x = self.conv2_2(x)
        # relu_conv2_2
        x = self.prelu2_2(x)
        # pool2 23->12
        x = self.avg_pool(x)
        # conv3_1 12->10
        x = self.conv3_1(x)
        # relu_conv3_1
        x = self.prelu3_1(x)
        # conv3_2 10->8
        x = self.conv3_2(x)
        # relu_conv3_2
        x = self.prelu3_2(x)
        # pool3 8->4
        x = self.avg_pool(x)
        # conv4_1 4->4
        x = self.conv4_1(x)
        # relu_conv4_1
        x = self.prelu4_1(x)
        # conv4_2 4->4
        x = self.conv4_2(x)
        # relu_conv4_2
        x = self.prelu4_2(x)
        # flatten
        result = x.view(-1, 4 * 4 * 80)
        # ip1
        result = self.ip1(result)
        # relu_ip1
        result = self.prelu_ip1(result)
        # ip2
        result = self.ip2(result)
        # relu_ip2
        result = self.prelu_ip2(result)
        # landmarks
        result = self.landmarks(result)
        return result


def main_test():
    # 参数设置部分
    # 创建解析器
    parser = argparse.ArgumentParser(description='Detector')
    # 添加参数
    # 批量训练集size
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    # 批量测试集size
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    # 训练epochs次数
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    # 学习率
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    # SGD动量
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    # 是否GPU不可用
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # 随机种子
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # 记录培训状态之前要等待多少批次
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    # 保存当前模型
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    # 模型保存路径
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    # Train/train, Predict/predict, Finetune/finetune
    parser.add_argument('--phase', type=str, default='Train',
                        help='training, predicting or finetuning')
    # 解析参数
    args = parser.parse_args()
    print(args.seed)


if __name__ == "__main__":
    main_test()
