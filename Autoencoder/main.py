import numpy as np
from utils.data_utils import get_mnist
from utils.plotting import plot_grid_of_images
import matplotlib.pyplot as plt

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

names = ['train_images', 'train_labels', 'test_images', 'test_labels']
sets = [train_images, train_labels, test_images, test_labels]

print("数据集导入成功～")

for i in range(len(sets)):
    print("_________________________________")
    print('[{} 的 类型是] {}'.format(names[i], type(sets[i])))
    print('[{} 的 形状是] {}'.format(names[i], sets[i].shape))
    print('[{} 的 数据类型是] {}'.format(names[i], sets[i].dtype))
    print("_________________________________")

print('训练集内 类别为: ')
print(np.unique(sets[1]))
print('测试集内 类别为: ')
print(np.unique(sets[3]))

plot_grid_of_images(train_images[0:100], n_imgs_per_row=10)
plt.show()