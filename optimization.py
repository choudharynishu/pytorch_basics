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
"""
# Define a Transformation pipeline - this will be provided as input to training and test set objects
#transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2861,), (0.3530,))])
transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2861,), (0.3530,))])
# Define training, validation, and test datasets
# Return type is Torchvision Dataset - like all datasets __getitem__ and __len__ methods are present
training_dataset = FashionMNIST(root=dataset_path, train=True, transform=transformation, download=True)
training_data, validation_data = data.random_split(training_dataset, lengths=[50000, 10000])

test_data = FashionMNIST(root=dataset_path, train=False, transform=transformation, download=True)

#Create Dataloaders
training_loader = DataLoader(training_data, batch_size=1024, shuffle=True, drop_last=False)
val_loader = DataLoader(validation_data, batch_size=1024, shuffle=False, drop_last=False)
test_loader = DataLoader(test_data, batch_size=1024, shuffle=False, drop_last=False)
# ---------------------------------------------------Neural Network Architecture-------------------------------------- #
class BaseNetwork(nn.Module):
    def __init__(self, activation_function, input_dim=784, output_dim=10, hidden_layers=[512, 256, 256, 128]):
        super.__init__()
        layers = []
        # ----First Layer
        layers += [nn.Linear(input_dim, hidden_layers[0]), activation_function]
        # ----Hidden Layers
        for i in range(1, len(hidden_layers)):
            layers += [nn.Linear(hidden_layers[i - 1], hidden_layers[i]), activation_function]

        # ----Output Layers
        layers+= [nn.Linear(hidden_layers[-1], output_dim)]

        self.layers = nn.Sequential(*layers)
        self.config = {'activation_func': activation_function.__class__name,
                       'input_size': input_dim,
                       'num_classes': output_dim,
                       'hidden_layers': hidden_layers
                       }

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Reshape the image to a flat vector
        out = self.layers(x)
        return out

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