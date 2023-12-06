import torch
import torch.nn as nn
import model.resource.dataset as dataset
import model.resource.mps_checker as checker


# 创建一个模型类
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 3)
        self.layer2 = nn.Linear(3, 1)

    def forward(self, x):
        # 1层
        x_1 = torch.relu(self.layer1(x))
        # 2层 (出结果)
        x_2 = self.layer2(x)
        return x_2


def display(model):
    for name, amount in model.state_dict().items():
        print("参数{} ->\n 数值：{}".format(name, amount))
    print('\n')


def run():
    model, device = checker.check(LinearModel())
    display(model)


run()
