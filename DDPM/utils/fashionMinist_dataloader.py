# 开发者 haotian
# 开发时间: 2023/2/22 18:35

import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torch
import torch.nn as nn
import io
from PIL import Image

minist_train = torchvision.datasets.FashionMNIST(root='E:/BaiduNetdiskDownload/FashionMnist', train=True, download=True, transform=torchvision.transforms.ToTensor())