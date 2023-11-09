import numpy as np
import torch
import matplotlib.pyplot as plt

# Add you codes here (use x and y as variable names for input and tagets)
np.random.seed(42)
x = np.random.rand(100, 1)
y = 1 + 2 * x + 0.1 * np.random.randn(100, 1)

indx = np.arange(100)  # indices for all data points in x
print("Before shuffle: \n", indx)

print("打乱数组")
np.random.shuffle(indx)
print("After shuffle: \n", indx)

train_indx = indx[:80]  # first 80% for training
val_indx = indx[80:]  # remaining 20% for validation

# Generate inputs and targets for training step
x_train, y_train = x[train_indx], y[train_indx]
print(x_train.size, y_train.size)

# Generate inputs and targets for validation step
x_val, y_val = x[val_indx], y[val_indx]
print(x_val.size, y_val.size)

x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

x_val_tensor = torch.from_numpy(x_val).float()
y_val_tensor = torch.from_numpy(y_val).float()

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].scatter(x_train, y_train)  # plot the training dataset
axs[1].scatter(x_val, y_val)  # plot the validation dataset

import torch.nn as nn


class FirstModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1).float())
        self.b = nn.Parameter(torch.randn(1).float())

    def forward(self, x):
        return self.a + self.b * x


import torch.optim as optim

torch.manual_seed(42)

device = 'mps:0' if torch.backends.mps.is_available() else 'cpu'
print(f"The device used is: {device}\n")

# Now we can create a model
model = FirstModel().to(device)

for i in iter(model.parameters()):
    print(i)

print("\n")
# we can also inspect its parameters
print("Parameters before training: \n", model.state_dict())

# set learning rate
lr = 1e-1

# set number of epochs, i.e., number of times we iterate through the training set
epochs = 100

# We use mean square error (MSELoss)
loss_fn = nn.MSELoss(reduction='mean')

# We also use stochastic gradient descent (SGD) to update a and b
optimiser = optim.SGD(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()  # set the model to training mode
    optimiser.zero_grad()  # avoid accumulating gradients
    y_pred = model(x_train_tensor.to(device))
    loss = loss_fn(y_train_tensor.to(device), y_pred)
    loss.backward()  # calculate gradients
    optimiser.step()  # updates model's params

print("Parameters after training: \n", model.state_dict())
