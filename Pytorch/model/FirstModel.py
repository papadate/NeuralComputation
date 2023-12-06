def run():
    import torch
    import torch.optim as optim
    import model.resource.mps_checker as checker
    import model.resource.dataset as dataset

    class FirstModel(torch.nn.Module):
        def __init__(self):
            super(FirstModel, self).__init__()
            # 定义参数
            self.b = torch.nn.Parameter(torch.randn(1).float())  # 生成一个随机标量
            self.w = torch.nn.Parameter(torch.randn(1).float())  # 生成一个随机标量

        def forward(self, x):
            # formula: W * X + b
            return (self.w * x) + self.b

    def display(model):
        for name, amount in model.state_dict().items():
            print("参数{} -> 数值：{}".format(name, amount))
        print('\n')

    def training(model, device):
        # 设置超参数：
        # 学习率
        learning_rate = 1e-1  # 0.1
        # 设置循环整个训练集的次数
        epochs = 200
        # 去平均MSE损失
        loss_fn = torch.nn.MSELoss(reduction='mean')
        # 使用SGD优化参数
        optimiser = optim.SGD(model.parameters(), lr=learning_rate)

        # 准备数据
        '''
        0 -> x_train
        1 -> y-train
        2 -> x_val
        3 -> y_val
        '''
        torch_sets = dataset.gen_regression()

        # 开始循环
        for epoch in range(epochs):
            # 设置为训练模式 dropout / batch-normalization
            model.train()
            # 防止梯度累计， 清空上一次操作的梯度记录
            optimiser.zero_grad()
            # 利用目前的模型求出输出值
            Y_prediction = model(torch_sets[0].to(device))
            # 求损失， 用此轮输出值和正确目标值进行比较
            loss = loss_fn(torch_sets[1].to(device), Y_prediction)
            # 训练模式神经网络最终的输出是一个损失值。利用backpropagation计算每个参数的梯度
            loss.backward()
            # 优化参数 (梯度下降 原参数值 - 学习率*该参数斜率)
            optimiser.step()

    torch.manual_seed(42)

    model, device = checker.check(FirstModel())

    print("分析模型参数：")
    for i in iter(model.parameters()):
        print(i)

    print("训练前 参数为：")
    display(model)

    # 训练模型
    training(model, device)

    # 训练结束
    print("训练后 参数为：")
    display(model)

    choice = input("是否需要画图？[yes] or [no]\n")
    if choice == 'yes':
        import model.resource.plot as plot
        tensor_prediction = model(dataset.gen_regression()[0].to(device))
        numpy_prediction = tensor_prediction.detach().to('cpu').numpy()
        plot.draw_pots_line(numpy_prediction, dataset.gen_regression())
