import torch


def check(modelClass):
    if torch.backends.mps.is_available():
        device = "mps"
        print("检测到计算机可以使用：{}\n".format(device))
        model = modelClass.to(device)
        return model, device
    else:
        print("检测到计算机仅有CPU")
        model = modelClass
        return model, 'cpu'
