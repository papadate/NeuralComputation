import matplotlib.pyplot as plt


def draw_pots_line(prediction, torch_set):
    print('正在画图...')
    x_train = torch_set[0].numpy()
    y_train = torch_set[1].numpy()
    plt.scatter(x_train, y_train)
    plt.plot(x_train, prediction, label='Prediction Function', color='red')
    plt.grid(True)
    plt.legend()
    plt.show()


def draw_losses(losses):
    print("正在画图...")
    plt.plot(range(len(losses)), losses, label='Error Log')
    plt.grid(True)
    plt.legend()
    plt.show()


def draw_class(x, y):
    print("正在画图...")
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()