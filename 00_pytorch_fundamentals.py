# Resource notebook: https://www.learnpytorch.io/00_pytorch_fundamentals/

# Imports needed

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====================================================================================================
# Introduction to Tensors
# Creating tensors
# PyTorch tensors are created using torch.Tensor() - https://pytorch.org/docs/stable/tensors.html

scalar = torch.tensor(7)
vector = torch.tensor([7, 7])
matrix = torch.tensor([[7, 8], [9, 10], [11, 12]])
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])

# Commands:
# .ndim

scalar.ndim # shows how many elements includes the scope. In this case - 0
matrix.shape # shows what are the columns and rows if it's matrix, if it's TENSOR with more elemtns,
# then add more information => matrix - torch.size([3, 2]), TENSOR - torch.size([1, 3, 3])

# ====================================================================================================
# Random tensors
# Why random tensors?
#
# Random tensors are important because the way many neural networks learn is that they start with tensors full of
# random numbers and then adjust those random numbers to better represent the data.

# Create a random tensor of size/shape (3, 4)
random_tensor = torch.rand(3, 4)

# Create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(224, 224, 3)) # height, width, colour channels

# Zeros and ones

# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))

# Create a tensor of all ones
ones = torch.ones(size=(3, 4))

# Type of tensor
torch.dtype

# ====================================================================================================
# Creating a range of tensors and tensors-like

# Use torch.range()
torch.arange(start=0, end=10, step=2)

# Creating tensors like
one_to_ten = torch.arange(start=1, end=11, step=1)
ten_zeros = torch.zeros_like(input=one_to_ten)

# ====================================================================================================
# Tensor datatypes
# Note: The most common errors which we receive into with PyTorch and Deep Learing:
#
# Tensors not right datatype
# Tensors not right shape
# Tensors not on the right device

float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                              dtype=None, # what datatype is the tensor(e.g. float32 or float16)
                               device=None, # What device is your tensor on (CPU/CUDA, etc)
                               requires_grad=False) # Whether or not to track gradients with this tensors operations

float_32_tensor.dtype
float_16_tensor = float_32_tensor.type(torch.float16)
float_16_tensor

# ====================================================================================================
# Getting information from tensors (Tensor attributes)
# For datatype - use tensor.dtype
# For shape - use tensor.shape
# For device - use tensor.device

some_tensor = torch.rand(3, 4)
some_tensor

print(some_tensor)
print(f"Datatype: {some_tensor.dtype}")
print(f"Shape: {some_tensor.shape}")
print(f"Device tensor is on: {some_tensor.device}")