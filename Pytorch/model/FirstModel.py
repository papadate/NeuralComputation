import torch
import torch.optim as optim
import model.resource.mps_checker as checker
import model.resource.dataset as dataset


class FirstModel(torch.nn.Module):
    def __init__(self):
        super(FirstModel, self).__init__()
        # 定义参数
        self.b = torch.nn.Parameter(torch.randn(1).float())  # 生成一个随机标量
        self.w = torch.nn.Parameter(torch.randn(1).float())  # 生成一个随机标量

    def forward(self, x):
        # formula: W * X + b
        return (self.w * x) + self.b

def display(model):
    for name, amount in model.state_dict().items():
        print("参数{} -> 数值：{}".format(name, amount))

def training(model):
    # 设置超参数：
    # 学习率
    learning_rate = 1e-1  # 0.1
    # 设置循环整个训练集的次数
    epochs = 100
    # 去平均MSE损失
    loss_fn = torch.nn.MSELoss(reduction='mean')
    # 使用SGD优化参数
    optimiser = optim.SGD(model.parameters(), lr=learning_rate)

    # 准备数据
    torch_sets = da

    # 开始循环
    for epoch in range(epochs):
        # 设置为训练模式 dropout / batch-normalization
        model.train()
        # 防止梯度累计， 清空上一次操作的梯度记录
        optimiser.zero_grad()
        Y_prediction = model()


torch.manual_seed(42)

device = ""
model = checker.check(FirstModel())

print("分析模型参数：")
for i in iter(model.parameters()):
    print(i)

print("为训练前 参数为：")
display(model)

# 训练模型
training(model)