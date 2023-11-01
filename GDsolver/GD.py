import random
import warnings
import matplotlib.pyplot as plt
from sklearn import preprocessing  # for normalization
import pandas as pd
import numpy as np

print("**************************")
print("初始化中...")
print("波士顿城市房屋金额数据导入中...")
print("**************************")
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
boston_data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

data = boston_data
x_input = data  # a data matrix
y_target = target  # a vector for all outputs
# add a feature 1 to the dataset, then we do not need to consider the bias and weight separately
x_in = np.concatenate([np.ones([np.shape(x_input)[0], 1]), x_input], axis=1)
# we normalize the data so that each has regularity
x_in = preprocessing.normalize(x_in)

w1 = np.zeros(x_in.shape[1])  # for GDsolver
w2 = np.zeros(x_in.shape[1])  # for Mini_Batch_GDsolver
NumofIteration = 100000
eta = 1.3

row, col = np.shape(x_in)
print("已累计样本 {} 个".format(row))
print()


# function for calculate the target
def model(w, X):
    # w is a weight wector d+1 dimensions
    # X is a sample matrix n * (d+1) size -> n is the number of samples
    return np.dot(X, w)


def cost(w, X, y):
    residual = y - model(w, X)
    err = np.dot(residual, residual) / (2 * len(y))
    return err


# print(cost(w1, x_in, y_target))

def gradient(w, X, y):
    err = (np.dot(X, w)) - y
    return np.dot(X.T, err) / len(y)


def GDsolver(X, y, max_iteration, eta):
    w = np.zeros(X.shape[1])
    iterations = []
    errors = []
    print("GD 算法装载成功")
    print("-->GD 算法循环开始<--")
    for i in range(max_iteration):
        w = w - eta * (gradient(w, X, y))
        if (i % 10) == 0:
            # print("循环 {} 次，错误值是：{}".format(i, cost(w, X, y)))
            err_i = cost(w, X, y)
            iterations.append(i)
            errors.append(err_i)

    print("->GD 算法循环结束<-")
    return w, iterations, errors


w1, GD_iterations, GD_errors = GDsolver(x_in, y_target, NumofIteration, eta)

print()


def batchGDsolver(X, y, max_iteration, eta, batch):
    row, col = np.shape(X)
    w = np.zeros(X.shape[1])
    iterations = []
    errors = []
    samples = list(range(row))
    print("Mini-Batch GD 算法装载成功")
    print("-->Mini-Batch GD 算法循环开始<--")
    for i in range(max_iteration):
        # randomly pick up batch number of samples from all sample list
        sample_batch = random.sample(samples, batch)
        # extract x y info
        # list comprehension [pick all row belonging to 'sample_batch', pick up all columns from selected row]
        sample_batch_X = X[sample_batch, :]
        sample_batch_y = y[sample_batch]

        # update weight vector
        w = w - eta * (gradient(w, sample_batch_X, sample_batch_y))

        # record segment
        if (i % 10) == 0:
            cost_i = cost(w, sample_batch_X, sample_batch_y)
            # print("循环 {} 次，错误值是：{}".format(i, cost_i))
            iterations.append(i)
            errors.append(cost_i)

    print("->Mini-Batch GD 算法循环结束<-")
    return w, iterations, errors


w2, mini_GDiterations, mini_GDerrors = batchGDsolver(x_in, y_target, NumofIteration, eta, 100)

# plt drawing diagram
print("开始画图！")
plt.plot(mini_GDiterations, mini_GDerrors, label="Mini_batch_GD Error", linestyle="--", linewidth=0.1)
plt.title("Mini Batch GD solver")
plt.xlabel("Number of iterations")
plt.ylabel("Error at this iteration")

plt.plot(GD_iterations, GD_errors, label="GD Error")
plt.title("GD Solver")
plt.xlabel("Number of iterations")
plt.ylabel("Error at this iteration")

plt.legend()
plt.grid(True)

plt.show()
