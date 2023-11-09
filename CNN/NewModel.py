import torch.nn as nn
import torch
import data_maker
import numpy as np
import matplotlib.pyplot as plt


class NewModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 添加线性层 输入 1维 输出 1维
        self.linear1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear1(x)


# 检测设备是否支持GPU
device = 'mps:0' if torch.backends.mps.is_available() else 'cpu'
# 创建一个NewModel 实例
model = NewModel().to(device)
print("模型参数为：")
parameters = list(model.parameters())
for i in range(len(parameters)):
    print(parameters[i])
# print(model.state_dict())
print("模型信息为：")
print(model)

# 生成数据
size = 100


# 生成数据函数

def myfun(x, NumberofInput):
    y = 1 + 2 * x + 40 * np.random.rand(NumberofInput, 1)
    return y


# 画图函数
def draw(x, y):
    plt.scatter(x, y)
    plt.grid(True)
    plt.show()


# 生成实例
data_maker1 = data_maker.data(size)
x, y = data_maker1.gen_data(myfun)
print(x.size)
print(y.size)

draw(x, y)
