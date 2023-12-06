def run():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np
    import model.resource.mps_checker as checker

    def generic_code(model, tensors, loss_fn, optimiser, numberOfSample, batch_size):
        def train_step():
            optimiser.zero_grad()
            random_choice = np.random.choice(numberOfSample, batch_size, replace=False)
            randomX = (tensors[0])[0, random_choice, :]
            randomy= (tensors[1])[0, random_choice]
            output = model(randomX)
            loss = loss_fn(randomy.long(), output)
            loss.backward()
            optimiser.step()

            return loss.items(), output

        return train_step

    def training(model, tensors, device, numberOfSample):
        loss_fn = nn.CrossEntropyLoss()
        optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        batch_size = 1600
        epochs = 1000

        tensors = [tensors[0].to(device), tensors[1].to(device)]

        train_step = generic_code(model, tensors, loss_fn, optimiser, numberOfSample, batch_size)

        losses = []
        outputs = []
        for epoch in range(epochs):
            loss, output = train_step()
            losses.append(loss)
            outputs.append(output)

        print("训练结束！")
        predictions = outputs[-1].max(1)[1]
        print(predictions)

    class Classification_Linear_Model(nn.Module):
        def __init__(self):
            super(Classification_Linear_Model, self).__init__()

            # 模型层次
            # 2维 特征输入
            self.layer1 = nn.Linear(2, 100)
            self.layer2 = nn.Linear(100, 100)
            # 输出 他对样本 4 个类的结论。 更接近哪个类？
            self.layer3 = nn.Linear(100, 4)

        def forward(self, x):
            x_1 = torch.relu(self.layer1(x))  # relu 激活函数
            x_2 = torch.relu(self.layer2(x_1))  # relu 激活函数
            x_3 = self.layer3(x_2)

            return x_3

    import model.resource.dataset as dataset

    numpy_set = dataset.gen_classification()
    # X
    tensor_X = torch.from_numpy(numpy_set[0])
    # 提高一个维度
    tensor_X = tensor_X.unsqueeze(0)

    # y
    tensor_y = torch.from_numpy(numpy_set[1])
    tensor_y = tensor_y.unsqueeze(0)

    tensors = [tensor_X.float(), tensor_y.float()]

    # 样本数量
    numberOfSamples = tensor_X.shape[1]

    model, device = checker.check(Classification_Linear_Model())

    training(model, tensors, device, numberOfSamples)
