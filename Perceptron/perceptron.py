# import libraries
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# generate data
def generate_data(NumofSamples):
    X = np.zeros(shape=(NumofSamples, 2))
    y = np.zeros(NumofSamples)
    for i in range(NumofSamples):
        X[i, 0] = random.randint(0,20)
        X[i, 1] = random.randint(0,20)
        if (X[i,0] + X[i, 1]) > 20:
            y[i] = 1
        else:
            y[i] = 0

    return X, y
X, y = generate_data(100000)

print(X)
print(y)