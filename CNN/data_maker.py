import numpy as np
import torch


class data():
    def __init__(self, NumberofInput):
        self.NumberofInput = NumberofInput

    def gen_data(self, func):
        np.random.seed(42)
        x = np.random.uniform(-200, 150, (self.NumberofInput, 1))
        y = func(x, self.NumberofInput)
        return x, y

    def separate_set(self, x , y, percentage):
        NumberofTrain = int(self.NumberofInput * percentage)
        index = np.arange(self.NumberofInput)
        np.random.shuffle(index)
        index_group1 = index[:NumberofTrain]
        index_group2 = index[NumberofTrain:]
        x_train, y_train = x[index_group1], y[index_group1]
        x_val, y_val = x[index_group2], y[index_group2]
        return x_train, y_train, x_val, y_val

    def toTorch(self, set):
        torchSet = torch.from_numpy(set).float()
        return torchSet