import Model
import data
import matplotlib.pyplot as plt

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