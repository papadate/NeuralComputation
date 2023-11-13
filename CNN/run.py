import Model
import data
import matplotlib.pyplot as plt
import torch.optim as optim

# check availability of GPU on your device
device = data.mac_device_check()
print("目前设备计算使用:{}".format(device))
'''
create an instance of LinearModel
to.(device) will let all calculations on this model in device
when model instance established, 'require_grad' automatically True
'''
model = Model.LinearModel().to(device)

# print your model parameters
# data.print_parameters(model)

# generate sample data
# (size, dimension, func)
size = 100
dimension = 1
x_train, y_train = data.gen_data(size, dimension, data.func_linear)

# plt.scatter(x_train, y_train)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid(True)
# plt.show()

'''
after we get the dataset, we may use mini-batch to improve the performance
DataLoader can do this
DataLoader(dataset=some_dataset,
           batch_size=some_number,
           shuffle=True/False)
           # shuffle - 是否打乱
how to establish an instance of dataset
use TensorDataset !!!
TensorDataset(x_train(tensor_version), y_train(tensor_version))                      
'''
train_loader = data.gen_dataloader(x_train, y_train)

# hyperparameter setting
# learning rate
lr = 1e-1  # 0.1
# epochs
epochs = 100  # 100 times iteration
'''
mean square error as loss function
reduction based on the mean
mean - divide the sum of all loss by the number of samples
'''
loss_fn = Model.nn.MSELoss(reduction='mean')
# set up the optimiser
optimiser = optim.SGD(model.parameters(), lr=lr)

print("优化前：")
data.print_parameters(model)

print("开始训练：")
for epoch in range(epochs):
    # modify the model to training mode
    model.train()
    # read each batch from dataloader
    '''
    train_loader is an instance
    we established in the past
    '''
    for x_batch, y_batch in train_loader:

