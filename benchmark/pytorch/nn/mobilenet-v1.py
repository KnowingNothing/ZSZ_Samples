"""
MobileNet-v1 implemented in PyTorch
author: size zheng
"""
from __future__ import absolute_import

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch 
import torch.nn as nn 


class DepthwiseSeperableBlock(nn.Module):
    def __init__(self, depthwise_channel, pointwise_channel, down_sample=False, bias=False):
        super(DepthwiseSeperableBlock, self).__init__()
        stride = 2 if down_sample else 1
        self.conv1 = nn.Conv2d(depthwise_channel, depthwise_channel, 3, stride=stride, padding=1, bias=bias, groups=depthwise_channel)
        self.bn1 = nn.BatchNorm2d(depthwise_channel)
        self.relu = torch.relu
        self.conv2 = nn.Conv2d(depthwise_channel, pointwise_channel, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(pointwise_channel)

    def forward(self, inputs):
        ret = self.conv1(inputs)
        ret = self.bn1(ret)
        ret = self.relu(ret)
        ret = self.conv2(ret)
        ret = self.bn2(ret)
        ret = self.relu(ret)
        return ret 


class MobileNetV1(nn.Module):
    def __init__(self, num_class=1000, width_mult=1.0, bias=False):
        super(MobileNetV1, self).__init__()
        block = DepthwiseSeperableBlock
        def R(x):
            return round(x * width_mult)
        self.first = nn.Conv2d(3, R(32), 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(R(32))
        self.blocks = nn.Sequential(
                        block(R(32), R(64)),
                        block(R(64), R(128), down_sample=True),
                        block(R(128), R(128)),
                        block(R(128), R(256), down_sample=True),
                        block(R(256), R(256)),
                        block(R(256), R(512), down_sample=True),
                        block(R(512), R(512)),
                        block(R(512), R(512)),
                        block(R(512), R(512)),
                        block(R(512), R(512)),
                        block(R(512), R(512)),
                        block(R(512), R(1024), down_sample=True),
                        block(R(1024), R(1024)))
        self.pool = nn.AvgPool2d([7, 7])
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(R(1024), num_class, bias=False)
        self.pred = nn.Softmax(dim=1)

    def forward(self, inputs):
        ret = self.first(inputs)
        ret = self.bn1(ret)
        ret = self.blocks(ret)
        ret = self.pool(ret)
        ret = self.flatten(ret)
        ret = self.dense(ret)
        ret = self.pred(ret)
        return ret


if __name__ == "__main__":
    net = MobileNetV1()
    device = torch.device("cuda", 0)
    batch_size = 128
    trials = 400
    net.to(device)
    data_shape = (batch_size, 3, 224, 224)
    for p in net.parameters():
        if len(p.shape) < 2:
            nn.init.constant_(p, 0)
        else:
            nn.init.xavier_uniform_(p)

    inputs = torch.rand(size=data_shape, dtype=torch.float32)
    # warm up
    output = net(inputs.to(device))
    torch.cuda.synchronize()
    # real measure
    # sum_time = 0.0
    beg = time.time()
    for i in range(trials):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        output = net(inputs.to(device))
        # end.record()
        # torch.cuda.synchronize()
        # sum_time += start.elapsed_time(end)
    # print(output)
    end = time.time()
    print("end-to-end time cost=", (end - beg) * 1e3 / trials, "ms")
