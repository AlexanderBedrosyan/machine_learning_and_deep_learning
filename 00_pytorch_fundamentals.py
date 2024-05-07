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