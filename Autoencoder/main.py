import numpy as np
from utils.data_utils import get_mnist

# 常量参数
Data_direction = './data/mnist'
seed = 111111

download = False

# 数据集初始化
train_images, train_labels = [], []
test_images, test_labels = [], []

if download:
    train_images, train_labels = get_mnist(data_dir = Data_direction,
                                           train=True,
                                           download=True)
    test_images, test_labels = get_mnist(data_dir = Data_direction,
                                         train=False,
                                         download=True)
else:
    train_images, train_labels = get_mnist(data_dir = Data_direction,
                                           train=True,
                                           download=False)
    test_images, test_labels = get_mnist(data_dir = Data_direction,
                                         train=False,
                                         download=False)
print("数据集导入成功～")