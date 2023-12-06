def run():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    class Classification_Linear_Model(nn.Module):
        def __int__(self):
            super(Classification_Linear_Model, self).__init__()

            # 模型层次
            # 2维 特征输入
            self.layer1 = nn.Linear(2, 100)
            self.layer2 = nn.Linear(100, 100)
            # 输出 他对样本 4 个类的结论。 更接近哪个类？
            self.layer3 = nn.Linear(100, 4)

        def forward(self, x):

            x_1 = F.relu(self.layer1(x))  # relu 激活函数
            x_2 = F.relu(self.layer2(x_1))  # relu 激活函数
            x_3 = self.layer3(x_2)

            return x_3