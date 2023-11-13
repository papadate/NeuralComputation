import torch
# 引入神经网络库
import torch.nn as nn
# 引入优化方法库
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1).float())
        self.b = nn.Parameter(torch.randn(1).float())

        # 设置向前传播的函数， 你实际上不需调用他
    def forward(self, x):
        return self.a + self.b * x

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

# 对应 100 个样本的数据
index = np.arange(100)
print(index)
print("打乱数组中...")
np.random.shuffle(index)
print(index)

# 分割数据集
train_index = index[:80]  # 从 0 - 79
validation_index = index[80:]  # 从 79 - 99

# 分配数据到 训练集 和 验证集
x_train = x[train_index]
y_train = y[train_index]
print("训练集数量x-{}, y-{}".format(x_train.size, y_train.size))
x_val = x[validation_index]
y_val = y[validation_index]
print("验证集数量x-{}, y-{}".format(x_val.size, y_val.size))

# # 转化numpy矩阵 到 torch 矩阵
x_train_torch = torch.from_numpy(x_train).float()
y_train_torch = torch.from_numpy(y_train).float()
x_val_torch = torch.from_numpy(x_val).float()
y_val_torch = torch.from_numpy(y_val).float()
# x_train_torch = torch.from_numpy(x_train).float()
# y_train_torch = torch.from_numpy(y_train).float()
#
# x_val_torch = torch.from_numpy(x_val).float()
# y_val_torch = torch.from_numpy(y_val).float()

# 画图：
# 创建一张图，1行两列（同一行里2张图）
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].scatter(x_train, y_train)
axs[1].scatter(x_val, y_val)
# plt.show()

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
learning_rate = 1e-1  # 0.1
epochs = 1000  # 训练数据循环次数
loss_fn = nn.MSELoss(reduction='mean')
# 损失函数 MSE.py 用均值
optimiser = optim.SGD(model.parameters(), lr=learning_rate)
# 使用SGD优化参数， lr-学习率是learning-rate, 提前设置好的
# 这里SGD将会优化参数 a 和 b

for epoch in range(epochs):
    # 设置为训练模式
    model.train()
    # 在每次epoch训练时，梯度归零，防止梯度积累
    optimiser.zero_grad()
    # 前向传播， 用目前的模型和输入计算出 预测y
    y_prediction = model(x_train_torch.to(device))
    # 通过定义好的损失函数来计算损失 你的标准答案和模型预测值
    loss = loss_fn(y_train_torch.to(device), y_prediction)
    print("第{}次 : loss -> {}".format(epoch, loss))
    # 利用loss进行反向传播
    loss.backward()
    # 之前定义的优化方法，step() 开始利用梯度和学习率去更新参数
    optimiser.step()
    # print(model.state_dict())

# # set learning rate
# lr = 1e-1
#
# # set number of epochs, i.e., number of times we iterate through the training set
# epochs = 100
#
# # We use mean square error (MSELoss)
# loss_fn = nn.MSELoss(reduction='mean')
#
# # We also use stochastic gradient descent (SGD) to update a and b
# optimiser = optim.SGD(model.parameters(), lr=lr)
#
# for epoch in range(epochs):
#     model.train()  # set the model to training mode
#     optimiser.zero_grad()  # avoid accumulating gradients
#     y_pred = model(x_train_torch.to(device))
#     loss = loss_fn(y_train_torch.to(device), y_pred)
#     loss.backward()  # calculate gradients
#     optimiser.step()  # updates model's params
#
print("参数训练完: \n", model.state_dict())

y_final = model(x_train_torch.to(device))
y_finalnp = y_final.detach().cpu().numpy()
y_val_final = model(x_val_torch.to(device))
y_finalval = y_val_final.detach().cpu().numpy()
axs[0].plot(x_train, y_finalnp, label='test', color='red')
axs[1].plot(x_val, y_finalval, label='val', color='purple')
plt.show()