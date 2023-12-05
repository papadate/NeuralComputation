import matplotlib.pyplot as plt


def draw(prediction, torch_set):
    print('正在画图...')
    x_train = torch_set[0].numpy()
    y_train = torch_set[1].numpy()
    plt.scatter(x_train, y_train)
    plt.plot(x_train, prediction, label='预测函数', color='red')
    plt.grid(True)
    plt.show()
