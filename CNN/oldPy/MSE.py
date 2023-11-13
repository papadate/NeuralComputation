import torch
import torch.nn as nn

# 创建 MSE 损失函数，使用默认的 'mean' reduction
mse_loss_mean = nn.MSELoss()

# 创建 MSE 损失函数，使用 'sum' reduction
mse_loss_sum = nn.MSELoss(reduction='sum')

# 创建 MSE 损失函数，使用 'none' reduction
mse_loss_none = nn.MSELoss(reduction='none')

# 示例输入
input_tensor = torch.tensor([1.0, 2.0, 3.0])
target_tensor = torch.tensor([2.0, 2.5, 3.5])

# 计算损失
loss_mean = mse_loss_mean(input_tensor, target_tensor)
loss_sum = mse_loss_sum(input_tensor, target_tensor)
loss_none = mse_loss_none(input_tensor, target_tensor)

print("Mean Reduction Loss:", loss_mean.item())
print("Sum Reduction Loss:", loss_sum.item())
print("None Reduction Loss:", loss_none)
