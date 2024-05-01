"""
This script will examine the impact of various initialization and optimization techniques on Neural Networks.
The impact is demonstrated on FashionMNIST dataset
Nishu Choudhary
"""

# Standard imports
import os
import json
import math
import random
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
from torchvision.datasets import FashionMNIST
from torchvision import transforms

# Imports for visualization
import plotly.graph_objects as go
from tqdm import tqdm

# ---------------------------------------------------Reproducibility-------------------------------------------------- #
seed = 13  # preference for a prime number


def seeding_fn(seed):
    """
    :param seed: seed for pseudo random generator
    :return: None
    """
    # ------- Standard seeding operations
    # Seeding Numpy operations
    np.random.seed(seed)
    # Seeding PyTorch operations for all (both CPU and CUDA) devices
    torch.manual_seed(seed)
    # Seeding for other Python operations
    random.seed(seed)
    # ------- Additional operations for GPU operations:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Turn off CUDA convolutional benchmarking - ensures CUDA selects the same algorithm every time
        torch.backends.cudnn.benchmark = False
        # Ensures the (same) selected algorithm's non-deterministic parts are off
        torch.backends.cudnn.deterministic = True

        # Filling uninitialized memory
        torch.utils.deterministic.fill_uninitialized_memory = True

        # ------Later, while defining the Dataloader class
        # Preserving reproducibility for DataLoader class - using worker_init_fn() and generator
        """
        g = torch.Generator
        g.manual_seed(seed)
        
        DataLoader(dataset, 
                    batch_size=batch_size,
                    num_workers=num_workers,
                    worker_init_fn=seeding_fn,
                    generator=g)
        """
    return None


# ---------------------------------------------------Locations-------------------------------------------------------- #
# Dataset path - location to download the data
dataset_path = "./data"
# Path to store runs - checkpoint path
checkpoint_path = "./optimization/"
# Create checkpoint directory, exist_ok = True (don't raise FileExistError if directory exists
os.makedirs(checkpoint_path, exist_ok=True)
# ---------------------------------------------------Dataset---------------------------------------------------------- #
"""
The FashionMNIST dataset has 10 output classes with each data point containing a tuple of (PILimage, class_label). 
The image is of dimension 28 * 28 and should be converted to a tensor. The conversion of PIL image into a tensor 
can be done using an object created using transforms.ToTensor().
 
From documentation, 
transforms.ToTensor(): converts a PIL Image or ndarray (H*W*C) in the range of [0, 255] to a torch.FloatTensor 
(C * H * W).
"""
# ----- Data Transformation Pipeline
transform_pipeline = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.2861,), (0.3530,))])
# ----- Load the Dataset
train_dataset = FashionMNIST(root=dataset_path, train=True, transform=transform_pipeline, download=True)
test_dataset = FashionMNIST(root=dataset_path, train=False, transform=transform_pipeline, download=True)
# ----- Check the range of a random tensor from the training_dataset
print(f"Mean of the tensors in the Training is: {(train_dataset.data.float()).mean().item()}")
# ----- Divide the dataset into Training and Validation set
train_size = len(train_dataset)
train_length, val_length = int(0.8 * train_size), int(0.2 * train_size) # Divide data by 80-20 rule
train_set, validation_set = torch.utils.data.random_split(train_dataset, [train_length, val_length])
# ----- Create Dataloader, to iterate through the data batches
train_loader = data.DataLoader(train_set, batch_size=1024, shuffle=True)
validation_loader = data.DataLoader(validation_set, batch_size=1024, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=1024, shuffle=True)
# ---------------------------------------------------Neural Network Architecture-------------------------------------- #
class BaseNetwork(nn.Module):
    def __init__(self, activation_func=nn.Sigmoid(), input_size=784, output_classes=10, hidden_layers=[512, 256, 256, 128]):
        super().__init__()
        layers = []
        input_dims = [input_size]+hidden_layers
        output_dims = hidden_layers + [output_classes]
        for (input_dim, output_dim) in zip(input_dims, output_dims):
            layers += [nn.Linear(input_dim, output_dim), activation_func]
        self.layers = nn.Sequential(*layers)
        print(layers)

model = BaseNetwork()

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
