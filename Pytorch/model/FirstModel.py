import torch


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

import torch.optim as optim
import model.resource.mps_checker as checker

torch.manual_seed(42)

device = ""
model = checker.check(FirstModel())

print("分析模型参数：")
for i in iter(model.parameters()):
    print(i)

print("为训练前 参数为：")
display(model)
