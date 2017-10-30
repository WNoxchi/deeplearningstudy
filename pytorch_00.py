# http://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
# 2017-Oct-30 00:41
# WNixalo

import torch

# Construct a 5x3 matrix, uninitialized:
x = torch.Tensor(5, 3)
print(x)

# Construct a randomly initialized matrix:
x = torch.rand(5, 3)
print(x)

# Get its size:
print(x.size())

# NOTE: torch.Size is a tuple so it supports the same operations

# Addition: syntax1
y = torch.rand(5, 3)
print(x + y)

# Addition syntax2:
print(torch.add(x, y))

# Addition: giving an output tensor:
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

# Addition in place:
y.add_(x)
print(y)

# Can use standard NumPy-like idnexing:
print(x[:, -1])

# NUMPY BRIDGE
# The Torch Tensor and NumPy Array will share their underlying memory locations,
# and changing one will change the other.

# Converting Torch Tensor to NumPy Array:
a = torch.ones(5)
print(a)

b = a.numpy();
print(b)

a.add_(1);
print(a); print(b)

# Converting numpy Array to torch Tensor:
# -- changing the np array changes the torch tensor automatically
import numpy as np
a = np.ones(5); b = torch.from_numpy(a);
np.add(a, 1, out=a);
print(a); print(b)

# All Tensors on the CPU except a CharTensor support converting to NumPy & back.
# CUDA Tensors:
# Tensors can be moved onto GPU using the .cuda function
# -- let us run this cell only if CUDA is available



















#
