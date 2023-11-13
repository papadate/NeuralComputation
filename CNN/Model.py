import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input and output from this linear is a scalar
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

class nnModel(nn.Module):
    def __init__(self):
        super(nnModel, self).__init__()
        # layers
        self.lay1 = nn.Linear(2, 100)
        self.lay2 = nn.Linear(100, 100)
        self.lay3 = nn.Linear(100, 4)

    def forward(self, x):
        x = self.lay1(x)
        x = nn.functional.relu(x)
        x = self.lay2(x)
        x = nn.functional.relu(x)
        x = self.lay3(x)
        return x

