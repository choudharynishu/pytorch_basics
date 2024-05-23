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
seed = 42  # preference for a prime number
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

# ------------------------------------Downloading Pre-trained models-------------------------------------------------- #
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
    # Needs an instantiation function because of special parameter 'alpha', this parameter needs to be accessible
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
        layers += [nn.Linear(input_size, hidden_layers[0]), act_function]
        # ----Hidden Layers
        for i in range(1, len(hidden_layers)):
            layers += [nn.Linear(hidden_layers[i - 1], hidden_layers[i]), act_function]

        # ----Output Layer
        layers += [nn.Linear(hidden_layers[-1], noutput_classes)]
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


#  ---------Load the Dataset FashionMNIST----------------------  #
"""
Downloads the famous Fashion MNIST dataset. The dataset has 10 output classes with each data point containing a tuple of
(PILimage, class_label). 
The image is of dimension 28 * 28 and should be converted to a tensor. The conversion of PIL image into a tensor 
can be done using an object created using transforms.ToTensor().
 
From documentation, 
transforms.ToTensor(): converts a PIL Image or ndarray (H*W*C) in the range of [0, 255] to a torch.FloatTensor 
(C * H * W) in range of [0.0, 1.0].

Then this range can be transformed to [-1, 1] using transforms.Normalize(mean =0.5, std =0.5)
transforms.Compose(*)
"""
# Compose the transformation pipeline
transform_pipeline = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download the training and testing dataset
train_dataset = FashionMNIST(root=dataset_path, train=True, transform=transform_pipeline, download=True)
test_dataset = FashionMNIST(root=dataset_path, train=False, transform=transform_pipeline, download=True)

train_size = len(train_dataset)
train_length, val_length = int(0.8 * train_size), int(0.2 * train_size)

"""
Each observation in the training set consists of a tuple (image, label). The images are black and white, consists of
a single channel
"""
# Divide the training set into train and validation set
train_set, validation_set = data.random_split(train_dataset, [train_length, val_length])

# Create Dataloader, to iterate through the data batches
train_loader = data.DataLoader(train_set, batch_size=1024,
                               shuffle=True)  # This loader is bigger in size compared to small loader in visualize function
validation_loader = data.DataLoader(validation_set, batch_size=1024, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=1024, shuffle=True)


#  ---------Visualize the gradient----------------------  #
def visualize_gradient(network):
    """
    :param network: Object of class BaseNetwork
    :return: Plotly histogram (subplot) of estimated gradients for the weights for different activation functions through different layers
    """

    network.eval()  # turn off dropout or batch normalization
    # create a small dataloader - batchsize =128
    train_small_loader = data.DataLoader(train_set, batch_size=128, shuffle=True)
    # extract input image and related labels from the next batch
    image, label = next(iter(train_small_loader))

    # For visualization purposes only single training step will be performed with every function call
    # Zero out the gradients - such that estimated gradients in this step don't accumulate from previous step
    network.zero_grad()
    # Compute predicted labels - forward pass to compute loss
    predicted_label = network(image)
    # Compute loss value -
    loss = F.cross_entropy(predicted_label, label)
    # Backpropagation
    loss.backward()
    # Access the estimated gradient - only restricted to weights
    gradients = {name: params.grad.data.view(-1).clone().numpy() for name, params in network.named_parameters() if
                 'weight' in name}
    network.zero_grad()

    # Plotly
    trace_list = []
    for key in gradients:
        trace = go.Histogram(x=gradients[key], name=f'{key}')
        trace_list += [trace]

    return trace_list


# ---- Call the visualization function
# - add labels to the subplots
# - make font smaller
ncolumns = nhidden_layers = len(BaseNetwork.__init__.__defaults__[-1]) + 1  # number of hidden layers
nrows = len(act_fn_by_name)
figure2 = make_subplots(rows=nrows, cols=ncolumns)
curr_row = 1
for i, activation_fn_name in enumerate(act_fn_by_name):
    activation_function = act_fn_by_name[activation_fn_name]()

    network_actfn = BaseNetwork(act_function=activation_function)
    trace_list = visualize_gradient(network_actfn)  # Everytime a new network object is sent
    #print(trace_list)
    curr_col = 1
    for trace in trace_list:
        figure2.append_trace(trace, curr_row, curr_col)
        curr_col += 1
    curr_row += 1
    del trace, trace_list

figure2.update_layout(showlegend=False)
figure2.write_html("gradients_backpropagation.html")
"""
1. Estimated gradients for Sigmoid activation at the input layer are very small,however they are very large at output layer
2. For ReLU the value of gradients is zero after layer#2
3. 
"""


# ------------------------------------------Training--------------------------------------------- #
def train_model(network, model_name, max_epochs=50, stop_iteration=7, batch_size=256, overwrite=False):
    """
    Train a model on the training set of FashionMNIST
    :param network: BaseNetwork object
    :param model_name: (str) Name of the model, used for creating checkpoint
    :param max_spochs: Maximum number of epochs to train for
    :param stop_iteration: Number of iterations to stop training if no performance gain observed for validation set
    :param batch_size: size of batch used for training
    :param overwrite: Whether or not to overwrite saved checkpoint. Default=False, skip training
    :return: Trained
    """
    file_exists = os.path.isfile(_get_model_file(checkpoint_path, model_name))
    if file_exists and not overwrite:
        print("Model file already exists. Skipping training...")
    else:
        if file_exists:
            print("Model file exists, but will be overwritten...")

    # ---- Defining optimizer, loss function, and Data Loader
    optimizer = optim.SGD(network.parameters(), lr=1e-2, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()  # check why not F.cross_entropy

    # ---- Training

    for epoch in range(max_epochs):
        network.train()
        true_predictions, count = 0, 0  # To track accuracy
        for image, label in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            predictions = network(image)

            loss = loss_function(predictions, label)
            loss.backward()
            optimizer.step()

            true_predictions += (predictions.argmax(dim=-1) == label).sum()
            count += label.shape[0]

        train_accuracy = true_predictions/count

    # ---- Validation
    validation_accuracy = test_model(network, validation_loader)


def test_model(network, test_loader):
    network.eval()
    true_predictions, count =0.0, 0
    for image, label in tqdm(test_loader):
        with torch.no_grad():
            predictions = network(image)
            true_predictions += (predictions.argmax(dim=-1) == label).sum()
            count += label.shape[0]

    test_accuracy = true_predictions / count
    return test_accuracy
