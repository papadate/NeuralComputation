import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input and output from this linear is a scalar
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
