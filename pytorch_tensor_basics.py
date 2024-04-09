# Import Standard packages
import os
import math
import numpy as np
import time
from tqdm.notebook import tqdm

# Imports for Visualization
import seaborn as sns

sns.set()

import torch

# For reproducibility
torch.manual_seed(13)

# To resolve OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# This error was eventually solved by installing nomkl

# Introduction to Tensors
# Syntax torch.Tensor(shape)
# Tip: Read dimensions from the right side that will be helpful in broadcasting as well
x = torch.Tensor(5, 2, 1)
x = torch.arange(10)
print(x.size())
# -------------------------------------------------------------------------------------------------------------------- #
'''
Other Tensor initialization operations:
1. torch.Tensor(shape) will allocate memory for the specified tensor shape but the values of the tensor will be random
2. torch.zeros(shape) will initialize a tensor of specified shape with zeroes as values; similar to numpy.zeros()
3. torch.ones(shape) similar to numpy.ones(shape) creates tensors with specified shape and values as one
4. torch.rand(shape) similar to numpy.random.rand(d0, d1, d2..) creates tensor values which are uniformly sampled 
   from [0, 1]
5. torch.randn(shape) similar to numpy.random.normal(loc=0, scale =1, shape)
6. torch.arange(M) creates a tensor with values starting from specified start (default =0) with specified number of 
   steps (default =1) 
'''
# -------------------------------------------------------------------------------------------------------------------- #
'''
Important Tensor attributes:
Assuming name of the tensor object created: x
1. x.shape = gives the shape of the created tensor
2. x.size() = function to get the shape value 
3. x.requires_grad (boolean) tells whether or not gradient will be estimated for the given tensor (default =True)
4. x.grad = stores the estimated value of the gradient (default = None)
5. x.dtype = specify the data type that will be stored in the tensor (default = )
6. x.device = available device types ('cuda','cpu','mps')
'''
# -------------------------------------------------------------------------------------------------------------------- #
'''
Tensor mathematical operations
1. x.T = Transpose operation
2. x.add_(y): Adds a tensor y to tensor x and the new value is stored in-place
3. x.view(): reshaping a tensor to a new shape
4. x.permute(1,0): Swaps dimensions 0 and 1
5. torch.matmul(x,y): For matrix multiplication can also be written as x @ y, numpy.matmul(x,y) - supports broadcasting
6. torch.mm: matrix product but doesn't support broadcasting
7. torch.bmm(x, y): Performs matrix multiplication with support for batch dimension, e.g.,
   x.shape = (b, m, n) y.shape = (b, n, p), then the shape of the output vector turns out to be output.shape = (b, m ,p)
8.    
'''
# -------------------------------------------------------------------------------------------------------------------- #

gpu_available = torch.cuda.is_available()
print(f"Is GPU available? {gpu_available}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

# Create a tensor and push it to the device
x = torch.ones(2, 4, 5)
print(x)
x = x.to(device)
print(x)

# -------------------------------------------------------------------------------------------------------------------- #
'''
Tensor mathematical operations
1. torch.cuda.is_available(): Check whether we have access to CUDA is available or not
2. torch.cuda.device_count(): Count number of GPUs available
3. torch.cuda_get_device_name(): Returns the name of the GPU device with specified index
4. torch.cuda.FloatTensor(*sizes): constructs a float tensor on the GPU
5. torch.cuda.synchronize(device=None): "Wait for all kernels in all streams on a CUDA device to complete."
6. torch.cuda.manual_seed(seed): Sets the seed for generating random numbers on the current GPU
7. torch.cuda.memory_allocated(device=None): Returns the current GPU memory allocatedby tensors in bytes
8. torch.cuda.memory_reserved(device=None): Returns the current   
'''
# -------------------------------------------------------------------------------------------------------------------- #