import numpy as np
class data():
    def __init__(self, NumberofInput):
        self.NumberofInput = NumberofInput

    def gen_data(self, func):
        np.random.seed(42)
        x = np.random.uniform(-200, 150, (self.NumberofInput, 1))
        y = func(x, self.NumberofInput)
        return x, y
