import torch_practice
import torch
import numpy as np

print("初始化 device参数(用来记录GPU是否可用的参数)")
device = None
print(device,'\n')
print("Mac 系统的设备检测过程有些不一样")

print("如果torch.backends.mps.is_available()检测为True")
print("设置 device = 'mps")
print("如果torch.backends.mps.is_available()检测为False")
print("设置 device = 'cpu, 因为你的电脑没有可用GPU")
if (torch.backends.mps.is_available()):
    device = 'mps'
else:
    device = 'cpu'
print("检测后：", device, '\n')

print("当我们发现设备GPU可用后，我们要把torch的tensor矩阵挪到GPU上计算")
print("方法如下：\n")

if(torch.)