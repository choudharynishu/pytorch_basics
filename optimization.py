"""
This script will examine the impact of various initialization and optimization techniques on Neural Networks.
The impact is demonstrated on FashionMNIST dataset
Nishu Choudhary
"""

# Standard imports
import os
import json
import math
import numpy as np
import pandas as pd

# Imports for Neural Network
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import for optimization
import torch.optim as optim

# Import for Data
import torch.utils.data as data
from torch.utils.data import DataLoader

# Imports for visualization
import plotly.graph_objects as go
from tqdm import tqdm

# ---------------------------------------------------Reproducibility-------------------------------------------------- #
seed = 42  # preference for a prime number
np.random.seed(seed)

# PyTorch Documentation on reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(seed)  # to seed the random number generation for all (both CPU and CUDA) devices
# ---------------------------------------------------Locations-------------------------------------------------------- #
# ---------------------------------------------------Dataset---------------------------------------------------------- #
# ---------------------------------------------------Neural Network Architecture-------------------------------------- #
# ---------------------------------------------------Visualizations--------------------------------------------------- #
# ---------------------------------------------------Initialization Techniques---------------------------------------- #
# ------- Constant Initialization
# ------- Constant Variance
# ------- Xavier Initialization
# ------- Kaiming Initialization
# ---------------------------------------------------Optimization Techniques------------------------------------------ #
# ------- SGD Optimization
# ------- SGD Momentum
# ------- Adam
# ------- Kaiming Initialization
