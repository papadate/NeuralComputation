import Model
import data
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

"""
generate data
"""
import numpy as np
import torch
# Add you codes here (use x and y as variable names for input and tagets)
np.random.seed(42)
x = np.random.rand(100, 1)
y = 1 + 2 * x + 0.1 * np.random.randn(100, 1)

indx = np.arange(100)  # indices for all data points in x
np.random.shuffle(indx)
train_indx = indx[:80]  # first 80% for training
val_indx = indx[80:]    # remaining 20% for validation

# Generate inputs and targets for training step
x_train, y_train = x[train_indx], y[train_indx]

# Generate inputs and targets for validation step
x_val, y_val = x[val_indx], y[val_indx]

x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

x_val_tensor = torch.from_numpy(x_val).float()
y_val_tensor = torch.from_numpy(y_val).float()
"""
generation section ends
!!! !!!
"""

train_loader = data.gen_dataloader(x_train_tensor, y_train_tensor)

epochs = 100
lr = 1e-1
device = data.mac_device_check()

model = Model.LinearModel().to(device)
loss_func = nn.MSELoss(reduction='mean')
optimiser = optim.SGD(model.parameters(), lr=lr)

def generic_code(model, loss_fn, optimiser):

    # define a function inside another function
    def train_step(x_batch, y_batch):
        optimiser.zero_grad()
        y_pred = model(x_batch)   # forward pass
        loss = loss_fn(y_batch, y_pred)  # calculate loss value
        loss.backward()                # autograd
        optimiser.step()               # update parameters
        return loss.item()             # return the loss

    # return the newly defined function
    return train_step                # return a function

print("优化前：")
data.print_parameters(model)

print("开始训练：")
train_step = generic_code(model, loss_func, optimiser)
# x_train_tensor = x_train_tensor.to(device)
# y_train_tensor = y_train_tensor.to(device)
losses = list()
for epoch in range(epochs):
    # modify the model to training mode
    model.train()
    # read each batch from dataloader
    '''
    train_loader is an instance
    we established in the past
    '''

    for x_batch, y_batch in train_loader:
        # # 注意 此时 batch的运算已经在GPU上了
        # x_batch = x_batch.to(device)
        # y_batch = y_batch.to(device)
        # # clean the gradient list
        # optimiser.zero_grad()
        # # calculate prediction values
        # y_pred = model(x_batch)
        # loss = loss_func(y_batch, y_pred)
        # loss.backward()
        # optimiser.step()
        losses.append(train_step(x_batch.to(device), y_batch.to(device)))

print("优化后：")
data.print_parameters(model)

"""
drawing section starts
"""
model.eval()
y = model(x_train_tensor.to(device)).detach().to('cpu').numpy()
# plt.plot(x_train, y)
# plt.scatter(x_train, y_train)
# plt.xlabel('x')
# plt.ylabel('prediction')
# plt.grid(True)
# plt.show()

plt.plot(range(len(losses)), losses, label="Training loss")
plt.xlabel("Training iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()