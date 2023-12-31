import numpy as np
import torch
import math
import model.resource.plot as plot


def gen_regression():
    # 固定随机种子
    np.random.seed(42)
    # 因为前面已经固定了随机种子，所以随机出来的数值永远每次执行任务都是一致的
    data_x = np.random.rand(100, 1)
    # 生成 对应输出 y 的值
    # 公式：y = b(b=1) + w(w=2) * x + (noise)
    data_y = 1 + 2 * data_x + 0.1 * np.random.randn(100, 1)

    index = np.arange(100)
    # print("打乱数组前：\n", index, '\n')
    print("打乱数组中... ...\n")
    np.random.shuffle(index)
    # print("打乱数组后： \n", index, '\n')

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
        # print(torch_sets[i])
        # print(torch_sets[i].dtype, "\n")
    print('\n')

    return torch_sets


def gen_classification():
    # 每个类点的数量
    pointsPerClass = 400
    # 每个样本数据的特征维度
    dimension = 2
    # 类的数量
    numberOfClass = 4

    # 设置 输入数据矩阵
    # 矩阵尺寸：    (行，                             列)
    #              >> 每个类多少个点 * 多少个类        >> 每个样本的特征维度
    X = np.zeros(((pointsPerClass * numberOfClass), dimension))
    # 设置 输出数据矩阵
    # 每个样本的输出 对应着 他所在的类
    # uint8 是一个 0 - 255  的正整数类型
    y = np.zeros((pointsPerClass * numberOfClass), dtype='uint8')

    # 循环每个类 设置x y
    for i in range(numberOfClass):
        '''
        0 -> 0 - 399
        1 -> 400 - 799
        ... ...
        '''
        X_singleClass = range(pointsPerClass* (i), pointsPerClass * (i+1))

        # 设置半径
        '''
        np.linspace()
        0.0 起点
        1 终点
        中间平均分割 pointsPerClass 份
        0～1 
        比如分出 5 个数
        1 2    3    4    5
        0 0.25 0.50 0.75 1
        '''
        radius = np.linspace(0.0, 1, pointsPerClass)

        # 设置角度
        #                   给每个样本一个角度                       干扰值
        theta = np.linspace(i * 4, (i + 1) * 4, pointsPerClass) + np.random.randn(pointsPerClass) * 0.2

        # 组合 sin cos函数 为 x 样本的两个特征
        X[X_singleClass] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        # 设置 正确标签
        y[X_singleClass] = i

    # 绘制样本图像
    # plot.draw_class(X, y)

    return X, y
