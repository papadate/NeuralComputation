import numpy as np
import math
import matplotlib.pyplot as plt

"""
生成数据代码段 - 开始
"""
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
y = np.zeros(N*K, dtype='uint8')

# 循环每一个类
for j in range(K):
    """
    j = 1
    range N -> 2N
    j = 2
    range 2N -> 3N
    j = 3
    range 3N -> 4N
    """
    ix = range(N * j, N * (j + 1))
    """
    np.linspace()
    可用参数：
    start 起始值
    stop 结束值
    num 中间要生成的等间隔样本 默认 50个
    dtype 输出数组类型
    axis 沿着哪个轴填充数组 默认 0
    0 -> 列的方向 竖着的
    1 -> 行的方向 横着的
    
    此代码行意义：
    从 0.0 开始 到 1 等间隔分割N个
    """
    r = np.linspace(0.0, 1, N) # 半径的数组
    #   角度数组                              随机噪声
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
    # 极坐标系 转换成 直接坐标系 （x， y）
    X[ix] = np.c_[r * np.cos(t), r * np.sin(t)]
    # 对 ix 数据 进行分类
    y[ix] = j

plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral)

"""
生成数据代码段 - 结束
"""

