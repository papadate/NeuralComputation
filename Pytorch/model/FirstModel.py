import torch


class FirstModel(torch.nn.Module):
    def __int__(self):
        super().__int__()
        # 定义参数
        self.b = torch.nn.Parameter(torch.randn(1).float())  # 生成一个随机标量
        self.w = torch.nn.Parameter(torch.randn(1).float())  # 生成一个随机标量

    def forward(self, x):
        # formula: W * X + b
        return (self.w * x) + self.b
