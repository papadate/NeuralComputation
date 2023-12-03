# import packages
import numpy as np
import torch

print("Construct a 3*4 randomly initialised matrix")
matrix = torch.rand(3,4)
print(matrix)
print("matrix's size is :", matrix.size(), "\n")

print("matrix.shape also works!")
print("matrix's size is :", matrix.shape, "\n")

print("You can construct a tensor directly from data")
print("ie. A List of Object")
print("[3, 4.5]")
list = [3, 4.5]
matrix = torch.tensor(list)
print(matrix)
print("matrix's size is :", matrix.shape, "\n")