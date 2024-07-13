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
from scipy import stats
import plotly.graph_objects as go
from tqdm import tqdm
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

sns.set()
# ---------------------------------------------------Reproducibility-------------------------------------------------- #
seed = 42  # preference for a prime number


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
checkpoint_path = "../optimization/"
# Create checkpoint directory, exist_ok = True (don't raise FileExistError if directory exists
os.makedirs(checkpoint_path, exist_ok=True)
# ---------------------------------------------------Dataset---------------------------------------------------------- #
"""
The FashionMNIST dataset has 10 output classes with each data point containing a tuple of (PILimage, class_label). 
The image is of dimension 28 * 28 and should be converted to a tensor. The conversion of PIL image into a tensor 
can be done using an object created using transforms.ToTensor().
"""
# Define a Transformation pipeline - this will be provided as input to training and test set objects
# transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2861,), (0.3530,))])
transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2861,), (0.3530,))])
# Define training, validation, and test datasets
# Return type is Torchvision Dataset - like all datasets __getitem__ and __len__ methods are present
training_dataset = FashionMNIST(root=dataset_path, train=True, transform=transformation, download=True)
training_data, validation_data = data.random_split(training_dataset, lengths=[50000, 10000])

test_data = FashionMNIST(root=dataset_path, train=False, transform=transformation, download=True)

# Create Dataloaders
training_loader = DataLoader(training_data, batch_size=1024, shuffle=True, drop_last=False)
val_loader = DataLoader(validation_data, batch_size=1024, shuffle=False, drop_last=False)
test_loader = DataLoader(test_data, batch_size=1024, shuffle=False, drop_last=False)


# ---------------------------------------------------Neural Network Architecture-------------------------------------- #
class BaseNetwork(nn.Module):
    def __init__(self, activation_function, input_dim=784, output_dim=10, hidden_layers=[512, 256, 256, 128]):
        super().__init__()
        layers = []
        # ----First Layer
        # Weights and biases in PyTorch are initialized using Kaiming Initialization (He initialization)
        layers += [nn.Linear(input_dim, hidden_layers[0]), activation_function]
        # ----Hidden Layers
        for i in range(1, len(hidden_layers)):
            layers += [nn.Linear(hidden_layers[i - 1], hidden_layers[i]), activation_function]

        # ----Output Layers
        layers += [nn.Linear(hidden_layers[-1], output_dim)]

        self.layers = nn.Sequential(*layers)
        self.config = {'activation_func': activation_function.__class__.__name__,
                       'input_size': input_dim,
                       'num_classes': output_dim,
                       'hidden_layers': hidden_layers
                       }

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Reshape the image to a flat vector
        # Since the BaseNetwork is defined using nn.Sequential in the forward pass we dont need to process each
        # layer using a loop. Alternatively, use following:
        # for l in self.layers:
        #       x = l(x)
        x = self.layers(x)
        return x


# ---------------------------------------------------Visualizations--------------------------------------------------- #

# --- Define Identity function as the activation function to demonstrate the impact of initialization
class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.config = {"name": self.name}

    def forward(self, x):
        return x


# --- Two visualizations are needed,
#       a. Visualize weights distribution inside the network
#       b. Visualize gradients that the parameters at different layers receive

# --- Visualization function - weights & Parameters
def plot_distributions(dict_val):
    """
    :param dict_val: Python dict with layer number as the key and values as either weights or gradient of the weights
    :param xlabel: (default  = 'weights') xlabel to be assigned to the plots
    :return: A Plotly subplot figure with each subplot representing weights of each layer
    """
    number_of_layers = len(dict_val)
    figure = make_subplots(1, number_of_layers)
    for index, (key, value) in enumerate(dict_val.items()):
        kernel_estimator = stats.gaussian_kde(value)
        kde_x = np.linspace(min(value), max(value), 1000)
        kde_y = kernel_estimator(kde_x)
        figure.add_trace(go.Histogram(x=value, nbinsx=50, name=f'Layer: {key}'), row=1, col=index + 1)
        figure.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines'), row=1, col=index + 1)
    return figure


# --- Function to access the weights of different layers
def weight_distribution(network):
    """
    Access the weights for each layer in form of a dictionary for the given network. The weights will be stored in form
    of a Python dictionary where the keys would represent the layer number while the values will store the parameter vector
    :param network: Object of class BaseNetwork whose weights need to be visualized
    :return: None (directly added to the histogram)
    """
    weights = {}
    for name, parameter in network.named_parameters():
        # Conditional to ignore the bias vectors
        if 'bias' in name:
            continue
        layer_number = name.split('.')[1]
        # detach() function returns a new tensor (storage?) removed from the computational graph, which implies
        # gradients for the detached tensor will not be tracked by the optimizer
        weights[layer_number] = parameter.detach().view(-1).numpy()

    weight_viz = plot_distributions(weights)
    weight_viz.show()
    return None


# ---Function to access the gradients associated with weights of each layer
def gradient_distribution(network):
    """
    Access the gradients of weights for each layer in form of a dictionary for the given network. The graidents will be
    stored in form of a Python dictionary where the keys would represent the layer number while
    the values will store the gradient value vector
    :param network: Object of class BaseNetwork whose gradients need to be visualized
    :return: None (directly added to the histogram)
    """
    gradients = {}

    # ---Generate gradients of the weights by single training step

    # Set the network to evaluation mode - turn off batch normalization or Dropouts
    network.eval()
    # Create a small training data loader - one can use previously created training_loader but its larger in size
    training_small_loader = DataLoader(training_data, batch_size=64, shuffle=True)
    # Iterate through the created dataloader
    train_features, train_labels = next(iter(training_small_loader))

    # ---Pass one batch through the network and estimate the gradients for the weights
    # Reset the gradients of all optimized torch tensors - clear out old gradients
    network.zero_grad()
    # Estimate predictions
    predicted_labels = network(train_features)
    # Estimate the loss function
    loss = F.cross_entropy(predicted_labels, train_labels)
    # Compute new gradients
    loss.backward()
    # Access the gradients
    for name, parameters in network.named_parameters():
        # Conditional to ignore the bias vectors
        if 'bias' in name:
            continue
        layer_number = name.split('.')[1]
        gradients[layer_number] = parameters.grad.data.view(-1).numpy()

    # Printing variance
    for key in sorted(gradients.keys()):
        print(f"Layer {key} gradient variance: {np.var(gradients[key])}")
    gradient_viz = plot_distributions(gradients)
    gradient_viz.show()


# --- Function to access activations
def activation_distribution(network):
    """
    Access the activations (output of the linear layers) in form of a dictionary for the given network.
    The activation values will be stored in form of a Python dictionary where the keys would represent the layer number
    while the values will store the activation value vector
    :param network: Object of class BaseNetwork whose activations need to be visualized
    :return: None (directly added to the histogram)
    """
    # Activations can be accessed using the forward hook
    activations = {}
    # Set the network to evaluation mode - turn off batch normalization or Dropouts
    network.eval()
    # Create a small training data loader - one can use previously created training_loader but its larger in size
    training_small_loader = DataLoader(training_data, batch_size=64, shuffle=True)
    # Iterate through the created dataloader
    train_features, train_labels = next(iter(training_small_loader))

    features = train_features.view(train_features.shape[0], -1)
    # Temporarily set all requires_grad flag to False - reduce memory usage and computational cost
    with torch.no_grad():
        for layer_index, layer in enumerate(network.layers):
            features = layer(features)
            if isinstance(layer, nn.Linear):
                activations[f"Layer {layer_index}"] = features.view(-1).detach().numpy()

    activation_viz = plot_distributions(activations)
    activation_viz.show()

    # Printing variance
    for key in sorted(activations.keys()):
        print(f"Layer {key} activation variance: {np.var(activations[key])}")

    return None


# ---------------------------------------------------Initialization Techniques---------------------------------------- #
# ------- Constant Initialization
def constant_initialization(network, constant_val=0.05):
    """
    :param network: (BaseNetwork) Object of class BaseNetwork whose parameters will be replaced by a constant value
    :param constant_val: (Float) Constant value for which all parameters (weights and biases) will be initialized
    :return: None
    """
    for name, parameters in network.named_parameters():
        parameters.data.fill_(constant_val)
    return None


# ------- Constant Variance
def constant_variance(network, constant_val=0.01):
    """
    :param network: (BaseNetwork) Object of class BaseNetwork whose parameters will be replaced by a constant value
    :param constant_val: (Float) Constant value for which all parameters' (weights and biases) variance be set to
    :return: None
    """
    for name, parameters in network.named_parameters():
        parameters.data.normal_(std=constant_val)
    return None


# def _bias_constant(constant_value=0): - can change the bias term later on


# ------- Xavier Initialization
def xavier_init(network):
    # Iterate over all weights and biases
    for name, parameter in network.named_parameters():
        # Initialize the bias term to be zero
        if name.endswith('.bias'):
            parameter.data.fill_(0)
        else:
            bound = math.sqrt(6 / (parameter.data.size(0) + parameter.data.size(1)))
            parameter.data.uniform_(-1 * bound, bound)


# ------- Kaiming Initialization
def kaiming_init(network):
    # Iterate over all weights and biases
    for name, parameter in network.named_parameters():
        # Initialize the bias term to be zero
        if name.endswith('.bias'):
            parameter.data.fill_(0)
        elif 'layer0' in name:

            parameter.data.normal_(0, std=1 / math.sqrt(parameter.data.size(1)))
        else:
            parameter.data.normal_(0, std=math.sqrt(2) / math.sqrt(parameter.data.size(1)))


test_network = BaseNetwork(activation_function=nn.ReLU())
# kaiming_init(network)
# weight_distribution(network)
# gradient_distribution(network)
# activation_distribution(network)

activation_fn_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "identity": Identity
}


# ---------------------------------------------------Training & Testing ---------------------------------------------- #
# ----------------------
def _get_config_file(model_path, model_name):
    return os.path.join(model_path, f"{model_name}.config")


def _get_model_file(model_path, model_name):
    return os.path.join(model_path, f"{model_name}.tar")


def _get_result_file(model_path, model_name):
    return os.path.join(model_path, f"{model_name}_results.json")


def load_model(model_path, model_name, net=None):
    """
    Load pre-trained model, if the model instance is not defined then instantiate the instance or else load the state dictionary
    :param model_path: (path-like object)
    :param model_name: (str)
    :param net: Object of class BaseNetwork
    :return: The loaded model
    """
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    assert os.path.exists(config_file), f"Path doesn't exist:\n {config_file}\n"
    assert os.path.exists(model_file), f"Path doesn't exist:\n {model_file}\n"
    # --- Read the config file
    with open(config_file, 'r') as f:
        config_dictionary = json.load(f)

    # --- Define the network object using config dictionary
    if net is None:
        act_function = config_dictionary['activation_func'].pop('name').lower()()
        net = BaseNetwork(activation_function=act_function, **config_dictionary)

    net.load_state_dict(torch.load(model_file))
    return net


def save_model(net, model_path, model_name):
    """
    Save configuration and model files
    :param net: Object of class BaseNetwork
    :param model_path: (path-like object)
    :param model_name: (str)
    """
    config_dict = net.config
    # --- Create directory, dont raise error if the directory already exists
    os.makedirs(model_path, exist_ok=True)
    # --- Extract configuration and model files path
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    # --- Save configuration dictionary into configuration files
    with open(config_file, 'w') as f:
        json.dump(config_dict, f)

    # --- Save model file into a state dictionary
    torch.save(net.state_dict(), model_file)


# ----------------------

def training(net, model_name, optim_function, max_epochs=50, batch_size=256, overwrite=False):
    """
    Training function
    :param net: Object of class BaseNetwork
    :param model_name: (str) name of the
    :param optim_function: (torch.optim.Optimizer class object)
    :param max_epochs: (int/float) default = 50
    :param batch_size: (int) default = 256
    :param overwrite: (bool) default = False
    :return:
    """
    # Check for results file already existing - if already exists and overwrite is False - skip training load results
    # Else check if file exists print if file already exists
    file_exists = os.path.isfile(_get_model_file(checkpoint_path, model_name))
    if file_exists and not overwrite:
        print(f"Model file exists. Skipping training")
        with open(_get_result_file(checkpoint_path, model_name), 'r') as f:
            results = json.load(f)
    else:
        if file_exists:
            print(f"Model file exists but will be overwritten")

        # --- Define optimizer, loss function, and use previously define dataloaders
        optimizer = optim_function(net.parameters())
        print(optimizer)

        loss_function = nn.CrossEntropyLoss()

        results = None
        validation_accuracy = []
        train_loss, train_accuracy = [], []
        # To keep track of best validation score  epoch
        best_val_epoch = -1
        # --- Training
        for epoch in tqdm(range(max_epochs)):
            net.train()  # --- Batch normalization and Dropout layers would be on
            true_predictions, total_observations = 0, 0
            current_epoch = tqdm(training_loader, leave=False)
            # -- Iterate through the Training DataLoader
            for image, label in current_epoch:
                # -- zero-out the gradient value
                optimizer.zero_grad()

                prediction = net(image)
                loss_value = loss_function(prediction, label)

                # -- Backpropagation
                loss_value.backward()

                # -- Update the parameters
                optimizer.step()

                # -- Statistics of this batch
                train_loss.append(loss_value.item())
                true_predictions += (prediction.argmax(dim=-1) == label).sum().item()
                total_observations += (label.shape[0])

                current_epoch.set_description(f"Epoch: {epoch + 1}: loss={loss_value.item():4.2f}")
            train_accuracy.append(true_predictions / total_observations)

            # ---- Validation
            validation_acc = test_model(net, val_loader)
            validation_accuracy.append(validation_acc)
            # Keep track of best validation score -- and save that particular model - save model's state_dict()
            if len(validation_accuracy) == 1 or validation_acc >= validation_accuracy[best_val_epoch]:
                best_val_epoch = epoch
                print(f" ----- Saving new best performing model")
                save_model(net, checkpoint_path, model_name)

        # Saving Results
        if results is None:
            # Load the model
            load_model(checkpoint_path, model_name, net)
            test_acc = test_model(net, test_loader)
            results = {"test_acc": test_acc, "val_scores": validation_accuracy,
                       "train_losses": train_loss, "train_scores": train_accuracy}
            # Save results
            with open(_get_result_file(checkpoint_path, model_name), 'w') as f:
                json.dump(results, f)
        # Plotting Training & Validation results
        accuracy_figure = go.Figure(go.Scatter(x=list(range(1, max_epochs)), y=train_accuracy,
                                               mode='lines', name=f"Training Accuracy"))
        accuracy_figure.add_trace(go.Scatter(x=list(range(1, max_epochs)), y=validation_accuracy,
                                             mode='lines', name=f"Validation Accuracy"))
        accuracy_figure.update_layout(title='Dataset Samples',
                                      xaxis_title='Epcoh No.',
                                      yaxis_title='Accuracy',
                                      )

        accuracy_figure.write_html(os.path.join(checkpoint_path, 'Train_validate_accuracy.html'))

    return None


def test_model(net, data_loader):
    """
    Test the trained model
    :param net: Object of class BaseNetwork
    :param data_loader: Validation or the test set loader
    :return: Average accuracy over all batches
    """
    # Evaluation mode
    net.eval()

    true_pred, total_observations = 0, 0
    with torch.no_grad():
        for image, label in data_loader:
            pred = net(image)
            true_pred += (pred.argmax(dim=1) == label).sum().item()
            total_observations += (label.shape[0])

    test_accuracy = true_pred / total_observations
    return test_accuracy


# ---------------------------------------------------Optimization Techniques------------------------------------------ #
# --- tester code
# optimizer_function = optim.SGD
# training(test_network, f"sample_07042024", optim.SGD)

# ---------------------------------------------------Optimization Techniques------------------------------------------ #
class OptimizerTemplate:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            self.update_params(p)

    def update_params(self, p):
        return NotImplementedError


# ------- SGD Optimization
class SGD(OptimizerTemplate):
    def __init__(self, params, lr):
        super().__init__(params, lr)

    def update_params(self, p):
        p_update = -self.lr * p.grad
        p.add_(p_update)


# ------- SGD Momentum
class SGDMomentum(OptimizerTemplate):
    def __init__(self, params, lr, momentum):
        super(SGDMomentum, self).__init__(params, lr)
        self.momentum = momentum
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}

    def update_params(self, p):
        self.param_momentum[p] = self.momentum * self.param_momentum[p] + (1 - self.momentum) * p.grad
        p_update = -self.lr * self.param_momentum[p]
        p.add_(p_update)


# ------- RMSProp
class RMSProp(OptimizerTemplate):
    def __init__(self, params, lr, alpha=0.99, epsilon=1e-8):
        super(RMSProp, self).__init__(params, lr)
        self.alpha = alpha
        self.epsilon = epsilon
        self.param_velocity = {p: torch.zeros_like(p.data) for p in self.params}

    def update_params(self, p):
        self.param_velocity[p] = self.alpha * self.param_velocity[p] + (1 - self.alpha) * p.grad ** 2
        p_update = (-self.lr / (torch.sqrt(self.param_velocity[p]) + self.epsilon)) * p.grad

        p.add_(p_update)


# ------- Adam
class Adam(OptimizerTemplate):
    def __init__(self, params, lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super().__init__(params, lr)
        # Alternatively, can access through self.betas[0] = beta_1, self.betas[1] = beta_2
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iteration = {p: 0 for p in self.params}  # Need to keep track of the number of iterations passed so far
        self.param_velocity = {p: torch.zeros_like(p.data) for p in self.params}
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}

    def update_params(self, p):
        self.iteration[p] += 1
        self.param_momentum[p] = self.beta_1 * self.param_momentum[p] + (1 - self.beta_1) * p.grad
        self.param_velocity[p] = self.beta_2 * self.param_velocity[p] + (1 - self.beta_2) * p.grad * p.grad

        p_scaled_momentum = self.param_momentum[p] / (1 - self.beta_1 ** self.iteration[p])
        p_scaled_velocity = self.param_velocity[p] / (1 - self.beta_2 ** self.iteration[p])

        p_update = (-self.lr / (torch.sqrt(p_scaled_velocity) + self.epsilon)) * p_scaled_momentum

        p.add_(p_update)


# -------------------------------------------------Test Run----------------------------------------------------------- #
model_test = BaseNetwork(activation_function=nn.ReLU())
kaiming_init(model_test)
training(model_test, f"sample_07062024", lambda params: SGD(params, lr=1e-1),
         max_epochs=30, batch_size=256)

# -------------------------------------------------Loss Surfaces------------------------------------------------------ #
# --- Pathological Curve
"""These are the types of curves which have steep gradient in one direction while a flat surface in another. This type of 
loss surface makes it difficult to optimize """


def pathological_curve_loss(w1, w2):
    x1_loss = torch.tanh(w1) ** 2 + 0.01 * torch.abs(w1)  # Steep direction
    x2_loss = torch.sigmoid(w2)  # Flat surface
    return x1_loss + x2_loss


# ---Steep Optima Curve
def bivar_gaussian(w1, w2, x_mean=0.0, y_mean=0.0, x_sig=1.0, y_sig=1.0):
    norm = 1 / (2 * np.pi * x_sig * y_sig)
    x_exp = (-1 * (w1 - x_mean) ** 2) / (2 * x_sig ** 2)
    y_exp = (-1 * (w2 - y_mean) ** 2) / (2 * y_sig ** 2)
    return norm * torch.exp(x_exp + y_exp)


def comb_func(w1, w2):
    z = -bivar_gaussian(w1, w2, x_mean=1.0, y_mean=-0.5, x_sig=0.2, y_sig=0.2)
    z -= bivar_gaussian(w1, w2, x_mean=-1.0, y_mean=0.5, x_sig=0.2, y_sig=0.2)
    z -= bivar_gaussian(w1, w2, x_mean=-0.5, y_mean=-0.8, x_sig=0.2, y_sig=0.2)
    return z


def plot_curve(curve_fn, x_range=(-5, 5), y_range=(-5, 5), plot_3d=True, title="Pathological curvature"):
    # Initialize a Plotly figure
    figure = go.Figure()
    # Generate x and y tensors

    x_tensor = torch.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / 100)
    y_tensor = torch.arange(y_range[0], y_range[1], (y_range[1] - y_range[0]) / 100)
    # Create a meshgrid for 3D plot
    grid_x, grid_y = torch.meshgrid(x_tensor, y_tensor, indexing='xy')
    # Create the call to the function curve (could be pathological or steep curve)
    z_tensor = curve_fn(grid_x, grid_y)
    x_array, y_array, z_array = grid_x.numpy(), grid_y.numpy(), z_tensor.numpy()

    if plot_3d:
        figure.add_trace(go.Surface(z=z_array, x=x_array, y=y_array))
    else:
        figure.add_trace(go.Contour(z=z_array, x=x_tensor.numpy(), y=y_tensor.numpy()))
    figure.update_layout(title=f"{title}",
                         xaxis=dict(title=f"$w_1$"),
                         yaxis=dict(title=f"$w_2$")
                         )
    figure.show()
    return figure


# ------------------------------------------Visualize each training algorithm----------------------------------------- #


def train_curve(optimizer_function, loss_curve=pathological_curve_loss, num_updates=100, init=[5, 5]):
    """
    :param optimizer_function: Constructor of the optimizer function to be used. The optimizer Template needs on parameters
    :param loss_curve: Loss function (by default tanh*sigmoid pathological curve)
    :param num_updates: Number of updates (steps) optimizer going to take to update the parameters
    :param init: Initial values of the parameters
    :return:
    result_array: Numpy array of shape [number of updates, 3], where [t,:2] represents parameters while
    [t,2] represents value of the loss function at t^th update
    """
    weights = nn.Parameter(torch.FloatTensor(init), requires_grad=True)
    optimizer = optimizer_function([weights])
    result_array = []

    for t in range(num_updates):
        # -- zero-out the gradient value
        optimizer.zero_grad()
        loss_value = loss_curve(weights[0], weights[1])
        # print(f"Before---, {[weights[0].data.detach(), weights[1].detach(), loss_value.item()]}")

        # -- Backpropagation
        loss_value.backward()
        # -- Update the parameters
        optimizer.step()

        result_array.append(torch.cat([weights.data.detach(), loss_value.unsqueeze(dim=0).detach()], dim=0))

    return torch.stack(result_array, dim=0).numpy()


def plot_optimization_curves(loss_curve, sgd_lr=0.01, sgd_mom_lr=0.001, sgd_momentum=0.9, adam_lr=1e-3, init=[5, 5],
                             plot_title='Pathological Loss curve'):
    SGD_points = train_curve(lambda params: SGD(params, lr=sgd_lr), loss_curve=loss_curve, init=init)
    SGDMomentum_points = train_curve(lambda params: SGDMomentum(params, lr=sgd_mom_lr, momentum=sgd_momentum),
                                     loss_curve=loss_curve, init=init)
    Adam_points = train_curve(lambda params: Adam(params, lr=adam_lr),loss_curve=loss_curve, init=init)

    all_points = np.concatenate([SGD_points, SGDMomentum_points, Adam_points], axis=0)
    # For Steep optima curve when we use [0, 0] initialization the
    if loss_curve.__name__ == 'comb_func':
        fig = plot_curve(loss_curve,
                         x_range=(-2, 2), y_range=(-2, 2),
                         plot_3d=False, title=plot_title)

    else:
        fig = plot_curve(loss_curve,
                         x_range=(-np.absolute(all_points[:, 0]).max(), np.absolute(all_points[:, 0]).max()),
                         y_range=(-np.absolute(all_points[:, 1]).max(), np.absolute(all_points[:, 1]).max()),
                         plot_3d=False,
                         title=plot_title)


    fig.add_trace(
        go.Scatter(x=SGD_points[:, 0], y=SGD_points[:, 1], mode='lines+markers', marker_color='red', name='SGD'))
    fig.add_trace(
        go.Scatter(x=SGDMomentum_points[:, 0], y=SGDMomentum_points[:, 1], mode='lines+markers', marker_color='blue',
                   name='SGD_momentum'))
    fig.add_trace(
        go.Scatter(x=Adam_points[:, 0], y=Adam_points[:, 1], mode='lines+markers', marker_color='grey', name='Adam'))
    fig.update_layout(showlegend=False)
    fig.show()


plot_optimization_curves(pathological_curve_loss, sgd_lr=10, sgd_mom_lr=10, sgd_momentum=0.9, adam_lr=1,
                         plot_title='Pathological Loss curve')
plot_optimization_curves(comb_func, sgd_lr=0.5, sgd_mom_lr=1, sgd_momentum=0.9, adam_lr=0.2, init=[0, 0],
                         plot_title='Steep Optima')

# ------------------------------------------Visualize effective learning Rate----------------------------------------- #
