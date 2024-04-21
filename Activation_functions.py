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
import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.optim as optim

# Imports for visualization
from tqdm import tqdm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

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
    def __init__(self, exp_alpha=1.0):
        super().__init__()
        self.config["exp_alpha"] = exp_alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.config["exp_alpha"] * (torch.exp(x) - 1))


class Swish(activation_functions):

    def forward(self, x):
        return x / (1 + torch.exp(-x))


# All names in lowercase - to be used later
act_fn_by_name = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leakyrelu": LeakyReLU,
    "elu": ELU,
    "swish": Swish
}


# --------------------------------------------Visualizing Activation Functions---------------------------------------- #
def grad_estimate(activation_func, x):
    """
    Estimates gradient for a given activation function at a particular input value x
    :param activation_func: Activation function whose gradient is to be estimated
    :param x: input tensor
    :return: Estimated gradient tensor
    """
    # Clone the input tensor and turn on requires_grad=True
    x = x.clone().requires_grad_()
    z = activation_func(x)  # This implicitly calls activation_function.forward(x)
    z.sum().backward()
    return x.grad


# List all activation functions
activation_funcs = [activation_func() for activation_func in act_fn_by_name.values()]
activation_func_names = [activation_func_name for activation_func_name in act_fn_by_name.keys()]
print(activation_func_names)
x_tensor = torch.linspace(-5, 5, 1000)
nrows = len(activation_funcs) // 2
ncols = 2
figure = make_subplots(rows=nrows, cols=ncols)
curr_row = curr_col = 1

for i, (act_func, act_func_nam) in enumerate(zip(activation_funcs, activation_func_names)):
    y = act_func(x_tensor)
    y_grad = grad_estimate(act_func, x_tensor)
    x, y, y_grad = x_tensor.numpy(), y.numpy(), y_grad.numpy()
    figure.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"{act_func_nam}"), row=curr_row, col=curr_col)
    figure.add_trace(go.Scatter(x=x, y=y_grad, mode='lines', name=f'Grad-{act_func_nam}'), row=curr_row, col=curr_col)
    if curr_col == 2:
        curr_col = 1
        curr_row += 1
    else:
        curr_col += 1
figure.write_html('activation_gradient_plot.html')


# --------------------------------------------Experiment on FashionMNIST Dataset-------------------------------------- #

#  ---------Neural Network Architecture----------------------  #
class BaseNetwork(nn.Module):
    def __init__(self, act_function, input_size=784, noutput_classes=10, hidden_layers=[512, 256, 256, 128]):
        """
        :param act_function: Activation or Squashing functions
        :param input_dimension: Size of the input class, default =784
        :param n_output_classes: Number of classes to be categorized into, default MNIST =10
        :param hidden_layers: A list of integers specifying the hidden layer sizes in the NN, length of the list number of layers
        """
        super().__init__()
        layers = []
        # ----First Layer
        layers[0] = nn.Linear(input_size, hidden_layers[0])

        # ----Hidden Layers
        for i in range(1, len(hidden_layers)):
            layers += [nn.Linear(hidden_layers[i - 1], hidden_layers[i]), act_function]

        # ----Output Layer
        layers += nn.Linear(hidden_layers[-1], noutput_classes)
        self.layers = nn.Sequential(*layers)

        # ----Storing hyperparameters in a dictionary
        self.config = {"act_function": act_function.config, "input_size": input_size, "num_classes": noutput_classes,
                       "hidden_layers": hidden_layers}

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Reshape the image to a flat vector
        out = self.layers(x)
        return out


#  ---------General functions for loading and saving models----------------------  #
# ----Internal functions to be accessed by the model loading and saving functions
def _get_config_file(model_path, model_name):
    config_file_path = os.path.join(model_path, model_name + '.config')

    return config_file_path


def _get_model_file(model_path, model_name):
    # Expected formats for model stored ['.tar', '.pkl', '.joblib']
    # potential_paths = [os.path.join(model_path, model_name + f'.{ext}') for ext in ['tar', 'pkl', 'joblib']]
    # model_file = next((model_fp for model_fp in potential_paths if os.path.exists(model_fp)), None)
    # Edit later to include other file formats
    return os.path.join(model_path, model_name + ".tar")


# ---- Load the model
def load_model(model_path, model_name, network=None):
    """
    Loads the given model from local disk
    :param model_path: Path of the checkpoint directory
    :param model_name: Name of the model ()
    :param network: Load the state dictionary into this object or else create a new model
    :return: The Neural Network model (newly created or loaded with state dictionary)
    """
    config_file_path, model_file_path = _get_config_file(model_path, model_name), _get_model_file(model_path,
                                                                                                  model_name)
    assert os.path.isfile(config_file_path), f"File not found: \"{config_file_path}\"."

    # ---- Load the config dictionary - need to pass the activation function (stored as name) as a function
    with(open(config_file_path, 'r')) as f:
        config_dict = json.load(config_file_path)
    # ---- Create a new model
    if network is None:
        # Retrieve the name of activation function this config_dict uses variable names defined in BaseNetwork model
        activation_fn_name = config_dict['act_function'].pop('name').lower()

        # Call the activation function with saved values for hyperparameters of the activation function
        # For example, act_fun = act_fn_by_name[activation_fn_name](**config_dict['act_function'].pop())
        # is equivalent to calling ELU(exp_alpha=0.1) if activation function was ELU in original model config.dict
        act_fun = act_fn_by_name[activation_fn_name](**config_dict.pop("act_function"))

        # ---- Instantiate the BaseNetwork model
        # The model's config_dict doesnt contain any information about the activation function- popped during last step
        network = BaseNetwork(act_function=act_fun, **config_dict)
    # state_dict is a dictionary that maps each layer to its parameter tensor - stored at last checkpoint
    network.load_state_dict(torch.load(model_file_path))

    return network


# ---- Save the model
def save_model(network, model_path, model_name):
    """
    Save the network's configuration dictionary and state dictionary
    :param network: Neural Network model to save
    :param model_path: Absolute path to the directory to save the config_dict
    :param model_name: Name of the Network file
    :return: None
    """
    # Save the configuration in config dictionary
    config_dict = network.config

    # Create the required directory - if already exists do nothing
    os.makedirs(model_path, exist_ok=True)

    # Create required file path
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)

    with open(config_file, 'w') as f:
        json.dump(config_dict, f)

    torch.save(network.state_dict(), model_file)
    return None


#  ---------Load the dataset----------------------  #

# ---- Data transformation applied to each image
# ---- Load the training set - split it into training and validation set
# ---- Load the test set
# ---- Create the DataLoader
# ---- Visualize the dataset

#  ---------Visualize the gradient----------------------  #

# ---- Input - Network, model_name, maximum number of epochs, stop_threshold, batch_size (small), overwrite
# -------- Network.eval() - turn off and drop or batch normalization
# -------- Create a small dataloader
# -------- every call load a new batch from the training set
# -------- Pass one batch through training - single training loss.backward(), loss function defined inside

