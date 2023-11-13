import random

import torch
import numpy as np


def mac_device_check():
    if torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def print_parameters(model):
    # model.state_dict() -> all parameter dictionaries
    # .items() extract every pair in the dictionary
    # such as (name1, value1)
    # for loop can read every element by iterations
    for name, weight in model.state_dict().items():
        print(name, ': ', weight)


def gen_data(size, dimension, func):
    x_train = np.random.uniform(-10, 50, size=(size, dimension))
    y_train = func(x_train)
    return x_train, y_train


def func_linear(x):
    # initialize the weight & bias
    number_of_samples, dimension = x.shape
    weight = np.zeros(dimension)
    for idx in range(dimension):
        weight[idx] = np.random.randint(1, 10)
    bias = random.randint(2, 10)

    # calculate every sample's target value y
    y_train = np.zeros((number_of_samples, 1))
    for sample_index in range(number_of_samples):
        noise = random.uniform(-10, 10)
        y_train[sample_index] = np.dot(x[sample_index], weight) + bias + noise

    return y_train
