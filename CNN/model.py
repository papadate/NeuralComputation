import torch
import torchvision
import torchvision.transforms as transforms
import Function
import matplotlib.pyplot as plt
import numpy as np


def run():
    device = Function.mps_checker()
    print("当前程序使用 硬件： {}".format(device))

    '''
    准备工作：
    数据预处理工具：
    transforms.Compose 用于定义对图像进行的预处理操作序列
    
    内部有两个操作：
    transforms.ToTensor() 
    将图像转换成Pytorch的Tensor 矩阵
    
    transforms.Normalize() 
    这个操作对Tensor进行标准化处理。传递的参数是均值和标准差
    就是一个正常的归一化操作
    '''

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4  # 一批处理的输入数据数量
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0)

    classes = (
        'plane',
        'car',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    )

    def imshow(img):
        img = img/2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    data_iter = iter(trainloader)
    images, labels = data_iter.next()

    imshow(torchvision.utils.make_grid(images))

    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))