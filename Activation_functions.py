'''
This script demonstrates the impact of choice of Activation function on the Neural Network.
Most common types of activation functions include,
1. Rectified Linear Units (ReLU)
    (other generalizations include, Absolute value recitifications, Leaky ReLU, and Parameteric ReLU)
2. Logistic Sigmoid
3. Hyperbolic Tangent
(*less frequently used)
4. Radial Basis function (RBF)
5. Softplus
6. Hard tanh

'''

# Standard imports
import os
import json
import math
import random
import numpy as np

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Data imports
import torch.utils.data as data
import urllib.request
from urllib.error import HTTPError

import torch.optim as optim

# Imports for visualization
from tqdm import tqdm
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ---------------------------------------------------Reproducibility-------------------------------------------------- #
seed = 17  # preference for a prime number
np.random.seed(seed)

# PyTorch Documentation on reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(seed)  # to seed the random number generation for all (both CPU and CUDA) devices

'''
** Custom Python Operators **
random.seed(seed)

** Additional steps for reproducibility for GPU operations **
Common sources of nondeterminism in PyTorch
a. CUDA convolutional benchmarking: Multiple convolutional algorithms are run to find the fastest by performing
                                    convolutional operation, this benchmarking process should be turned off for reproducibility
                                    torch.backends.cudnn.benchmark=False
b. Nondeterministic Algorithm in PyTorch operations: torch.use_deterministic_algorithms() throws an error if an operation
                                                     is known to be nondeterministic, e.g., torch.Tensor.index_add()
c. Filling uninitialized memory: Usage of functions like torch.empty() and torch.Tensor.resize_() can return tensors 
                                (of a given shape) with uninitialized values which can introduce randomness.
                                torch.utils.deterministic.fill_uninitialized_memory = True
d. DataLoader: DataLoader will reseed workers following Randomness in multi-process data loading algorithm, need to use
               worker_init_fn() and generator to preserve reproducibility.
               
                                                                              
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms()=True
g = torch.Generator
g.manual_seed(seed)
               
DataLoader(dataset,
           batch_size=batch_size,
           num_workers=num_workers,
           worker_init_fn=seed_worker,
           generator=g)
'''


# -------------------------------------------------------Locations---------------------------------------------------- #
def check_path(path):
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(path):
        os.makedirs(path)

    return None


dataset_path = "./data"
checkpoint_path = "./activation_fns"

check_path(dataset_path)
check_path(checkpoint_path)

# -------------------------------------------------------Data Import-------------------------------------------------- #
base_url = 'https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial3/'
files_to_download = ["FashionMNIST_elu.config", "FashionMNIST_elu.tar",
                     "FashionMNIST_leakyrelu.config", "FashionMNIST_leakyrelu.tar",
                     "FashionMNIST_relu.config", "FashionMNIST_relu.tar",
                     "FashionMNIST_sigmoid.config", "FashionMNIST_sigmoid.tar",
                     "FashionMNIST_swish.config", "FashionMNIST_swish.tar",
                     "FashionMNIST_tanh.config", "FashionMNIST_tanh.tar"]

for file in tqdm(files_to_download):
    # File location if it already exists
    file_path = os.path.join(dataset_path, file)

    # Download the file if it doesn't exists
    if not os.path.isfile(file_path):
        file_url = f"{base_url}{file}"
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print("Something wrong with URL retrieval, please try to download directly from Google Drive folder")

# -------------------------------------------------Activation Functions----------------------------------------------- #
'''
PyTorch documentation on (nonlinear) Activation or Squashing functions:
https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
These functions are available both as modules as well as functions
As Modules
-torch
    |--torch.nn
        |-torch.nn.Sigmoid
        |-torch.nn.Tanh
        
As Functions
-torch
    |-torch.sigmoid
    |-torch.tanh
'''


# Defining using the torch.nn module
class activation_functions(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.config = {"name": self.name}


# Defining as functions
class Sigmoid(activation_functions):
    # Instantiation function will be automatically inherited from the activation_function class
    def forward(self, x):
        return 1 / (1 + torch.exp(-x))


class Tanh(activation_functions):
    # Instantiation function will be automatically inherited from the activation_function class
    def forward(self, x):
        return ((torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x)))


class ReLU(activation_functions):
    # Instantiation function will be automatically inherited from the activation_function class
    # Derivative of ReLU is computed using the subgradient method
    def forward(self, x):
        return x * (x > 0).float()


class LeakyReLU(activation_functions):
    # Needs an instantiation function because of special parameter 'alpha'
    # Default value of alpha is 0.1
    def __init__(self, alpha=0.1):
        super().__init__()
        self.config["alpha"] = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.config["alpha"] * x)


class ELU(activation_functions):
    # Needs an instantiation function because of special parameter 'alpha'
    # Default value of alpha is 1.0
    def __init__(self, alpha=1.0):
        super().__init__()
        self.config["exp_alpha"] = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.config["exp_alpha"] * (torch.exp(x) - 1))


class Swish(activation_functions):

    def forward(self, x):
        return x / (1 + torch.exp(-x))


act_fn_by_name = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leakyrelu": LeakyReLU,
    "elu": ELU,
    "swish": Swish
}

# --------------------------------------------Visualizing Activation Functions---------------------------------------- #

# --------------------------------------------Experiment on FashionMNIST Dataset-------------------------------------- #