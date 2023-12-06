import torch
import torch.nn as nn
import torch.optim as optim



# 创建一个模型类
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 1)
        self.layer2 = nn.Linear(1, 1)

    def forward(self, x):
        # 1层
        x_1 = torch.relu(self.layer1(x))
        # 2层 (出结果)
        x_2 = self.layer2(x)
        return x_2


def display(model):
    for name, amount in model.state_dict().items():
        print("参数{} ->\n数值：\n{}\n".format(name, amount))
    print('\n')


def train1(model, train_loader, device, torch_set):
    print("初始训练方法进行中... ...")
    print("训练前参数为：")
    display(model)

    # 设置超参数
    learning_rate = 1e-1
    epochs = 100
    loss_fn = nn.MSELoss(reduction='mean')
    optimiser = optim.SGD(model.parameters(), lr=learning_rate)

    # 训练过程
    for epoch in range(epochs):
        model.train()
        # train_loader 已经根据我们设置自动分号mini_batch
        for x_batch, y_batch in train_loader:
            # 确保都在同一硬件内运算
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimiser.zero_grad()

            y_prediction = model(x_batch)
            loss = loss_fn(y_batch, y_prediction)
            loss.backward()

            optimiser.step()

    print("训练结束！")
    display(model)

    choice = input("是否需要画图？[yes] or [no]\n")
    if choice == 'yes':
        import model.resource.plot as plot
        tensor_prediction = model(torch_set[0].to(device))
        numpy_prediction = tensor_prediction.detach().to('cpu').numpy()
        plot.draw_pots_line(numpy_prediction, torch_set)

def run():
    import model.resource.dataset as dataset
    import model.resource.mps_checker as checker

    # 模型准备阶段
    model, device = checker.check(LinearModel())

    # 数据准备阶段
    torch_set = dataset.gen()

    from torch.utils.data import TensorDataset, DataLoader

    train_data = TensorDataset(torch_set[0], torch_set[1])
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

    train1(model, train_loader, device, torch_set)
