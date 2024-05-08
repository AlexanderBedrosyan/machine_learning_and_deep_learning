# Resource notebook: https://www.learnpytorch.io/00_pytorch_fundamentals/

# Imports needed

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


# Random tensors
# Why random tensors>
#
# Random tensors are important becaause the way many neural networks learn is that they start with tensors full of
# random numbers and then adjust those random numbers to better represent the data.

# Create a random tensor of size/shape (3, 4)
random_tensor = torch.rand(3, 4)
random_tensor

# Create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(224, 224, 3)) # height, width, colour channels
random_image_size_tensor.shape, random_image_size_tensor.ndim