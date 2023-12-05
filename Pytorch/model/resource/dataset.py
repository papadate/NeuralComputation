import numpy as np
import torch

# 固定随机种子
np.random.seed(42)
# 因为前面已经固定了随机种子，所以随机出来的数值永远每次执行任务都是一致的
data_x = np.random.rand(100, 1)
