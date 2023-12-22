import numpy as np
from utils.data_utils import get_mnist
from utils.plotting import plot_grid_of_images
import matplotlib.pyplot as plt
from utils.plotting import plot_train_progress_1, plot_grids_of_images

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

i = 100
plot_grid_of_images(train_images[i:i+100], n_imgs_per_row=10)
plt.show()

# 此步操作，是用来把2D矩阵拉直变成1D向量
train_images_flat = train_images.reshape([train_images.shape[0], -1])
test_images_flat = test_images.reshape([test_images.shape[0], -1])

print("numpy array 前后变化：")
print('1 -> {}'.format(train_images.shape))
print('2 -> {}'.format(train_images_flat.shape))

# 数据分割函数
def get_random_batch(train_images, train_labels, batch_size, random_generator):
    indices = range(0, batch_size)
    indices = random_generator.randint(low=0,
                                       high=train_images[0],
                                       size=batch_size,
                                       dtype='int32')
    train_images_batch = train_images[indices]

    # 标签集合分流： 监督训练有标签， 无监督训练无标签
    if train_labels is not None:
        train_labels_batch = train_labels[indices]
    else:
        train_labels_batch = None
    return train_images_batch, train_labels_batch