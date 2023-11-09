import torch
import numpy as np
from torch import Tensor

print("检查设备GPU是否可用")
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(device)
print(torch.cuda.is_available())

print("\nMac operation to use GPU acceleration\n")
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    # requires_grad True就是开始使用forward pass 和 backward pass
    x = torch.ones(3, device=mps_device, requires_grad=True)
    print(x)
    print(torch.backends.mps.is_available())
    print(x.device)
    y = torch.add(x, 2)
    print(y)
    z = torch.add(y, 2)
    print(z)
    print(x.grad_fn)
    print(y.grad_fn)
    print(z.grad_fn)
    z = y*y*2
    print(z)
    out = z.mean()
    print(out)

    x = torch.randn(2, 2)
    print(x)
    print(x.requires_grad)
    x.requires_grad_(True)
    print(x.requires_grad)

else:
    print("MPS device not found.")

