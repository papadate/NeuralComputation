import numpy as np
import torch

# 固定随机种子
np.random.seed(42)
# 因为前面已经固定了随机种子，所以随机出来的数值永远每次执行任务都是一致的
data_x = np.random.rand(100, 1)
# 生成 对应输出 y 的值
# 公式：y = b(b=1) + w(w=2) * x + (noise)
data_y = 1 + 2 * data_x + 0.1 * np.random.randn(100, 1)

index = np.arange(100)
print("打乱数组前：\n", index, '\n')
print("打乱数组中... ...\n")
np.random.shuffle(index)
print("打乱数组后： \n", index, '\n')

# 分割 数据集： 0～79 training data / 80～99 validation data
train = index[:80]  # 提取0～79 不包括第80位
val = index[80:]  # 提取80～99 知道index数组末尾截止

print("分配中... ...")
x_train, y_train = data_x[train], data_y[train]
x_val, y_val = data_x[val], data_y[val]

sets = [x_train, y_train, x_val, y_val]
content = ["输入训练集", "输出训练集", "输入验证集", "输出验证集"]
for i in range(2):
    print("{} 数量为：{}".format(content[i * 2], sets[i * 2].size))
    print("{} 数量为：{}".format(content[(i * 2) + 1], sets[(i * 2) + 1].size))

# 转换 numpy矩阵 -> tensor矩阵
# 注意！在输入进神经网络时，数据需保持为 float32 格式， 使用 .float() 方法来转换
torch_sets = []
for i in range(4):
    torch_sets.append(torch.from_numpy(sets[i]).float())
    print(torch_sets[i])
    print(torch_sets[i].dtype, "\n")