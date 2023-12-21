import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 24

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size,
    shuffle=True, num_workers=0
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size,
    shuffle=False, num_workers=0
)

classes = ('飞机',
           '汽车',
           '鸟',
           '猫',
           '鹿',
           '狗',
           '青蛙',
           '马',
           '船',
           '卡车')


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


PATH = './cifar_net.pth'


def train_mode():
    dataiter = iter(trainloader)

    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    # print(' '.join('%5s\n' % classes[labels[j]] for j in range(batch_size)))

    Net = ImageNet().to(device)
    Net.load_state_dict(torch.load(PATH))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(Net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = Net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/2000))
                running_loss = 0.0

    print('训练结束')

    torch.save(Net.state_dict(), PATH)


def test_mode():
    dataiter = iter(testloader)
    images, labels = next(dataiter) #Selects a mini-batch and its labels

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('标准答案: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    Net = ImageNet().to(device)
    Net.load_state_dict(torch.load(PATH))

    images = images.to(device)
    labels = labels.to(device)
    outputs = Net(images)

    _, predicted = torch.max(outputs, 1) #Returns a tuple (max,max indicies), we only need the max indicies.

    print('预测：', ''.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = Net(images)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('每1万张图片预测准度: %d %%' % (
            100 * correct / total))

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = Net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("类 {:5s} 准确率是: {:.1f} %".format(classname,
                                                             accuracy))


print("1 -> 训练")
print("其他 -> 测试")
user_input = input("输入：")
if user_input == "1":
    train_mode()
else:
    test_mode()