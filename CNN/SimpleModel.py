import torch
# 引入神经网络库
import torch.nn as nn
# 引入优化方法库
import torch.optim as optim
import numpy as np


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1).float())
        self.b = nn.Parameter(torch.randn(1).float())

    # 设置向前传播的函数， 你实际上不需调用他
    def forward(self, x):
        return self.a + (self.b * x)


print("所以，当你有一个模型实例 model 时，"
      "你可以直接将输入数据传递给模型，"
      "模型会自动调用其 forward 方法，"
      "执行前向传播，并返回预测结果。"
      "这是一种更简便的方式，"
      "而不需要显式调用 forward 方法。")

# 生成数据集代码块：
# 固定随机生成器，保证模型重复性
np.random.seed(42)
'''
x - input
y - target
'''

# 生成x 数据 100 个x 样本
x = np.random.rand(100, 1)
# 定义计算y 的函数， 1 + 2倍的x + 一些干扰
y = 1 + (2 * x) + (0.1 * np.random.randn(100, 1))







# 设置固定随机数生成器，保证每次运时随机数一至
torch.manual_seed(42)
# 检查设备GPU是否可用， Mac的GPU要用 backends.mps.is_available() 检查
device = 'mps:0' if torch.backends.mps.is_available() else 'cpu'
print("GPU 是否可用：{}".format(torch.backends.mps.is_available()))
# 创建一个实例，并且确保使用GPU运算
model = SimpleModel().to(device)

# 遍历模型内参数： 并且打印每个参数
for i in iter(model.parameters()):
    print(i)

print("\n")
print("参数 训练之前：\n {}".format(model.state_dict()))

# 设置超参数：
learning_rate = 1e-1  # 0,1
epoches = 100 # 训练数据循环次数
loss_fn = nn.MSELoss(reduction='mean')
# 损失函数 MSE 用均值
optimiser = optim.SGD(model.parameters(), lr=learning_rate)
# 使用SGD优化参数， lr-学习率是learning-rate, 提前设置好的
# 这里SGD将会优化参数 a 和 b

for epoch in range(epoches):
    model.train()
    optimiser.zero_grad()
    y_prediction = model()
