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
data.print_parameters(model)

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

# hyperparameter setting
# learning rate
lr = 1e-1  # 0.1
# epochs
epochs = 100  # 100 times iteration
'''
mean square error as loss function
reduction based on the mean
'''

loss_fn = Model.nn.MSELoss(reduction='mean')


