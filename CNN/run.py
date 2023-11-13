import Model
import data

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


