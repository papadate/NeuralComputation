import numpy as np
import math
import matplotlib.pyplot as plt

N = 400  # number of points per class
D = 2  # dimension
K = 4  # number of class
#
"""
set up a input data matrix
row - samples
col - dimensions
*
sample = number of points each class (N) * number of class (K)
"""
X = np.zeros((N*K, D))
y = np.zeros(N*K, dtype='unit8')

# 循环每一个类
for j in range(K):
    """
    j = 1
    range N -> 2N
    j = 2
    """
    ix = range(N * j, N * (j + 1))
