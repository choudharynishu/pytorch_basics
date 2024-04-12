# Required Imports
import torch

# For defining the Neural Network Architecture
import torch.nn as nn
import torch.nn.functional as F

# For defining the Dataset class
import torch.utils.data as data

# For visualizations
import plotly.graph_objects as go
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------Neural Network Architecture-------------------------------------------------- #
class Classifier(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        '''
        Initialize the layers of the proposed Neural Network Architecture
        '''
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.activation_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        '''
        Here the computation of the module takes place, and is executed when one calls the module(nn=Classifier(); nn(x))
        This is where the user defines the architecture of the proposed Neural Network
        :param x: Input values to the Neural Network architecture
        :return: Estimated output values (vector) computed
        '''
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


model = Classifier(num_inputs=2, num_hidden=4, num_outputs=1)
print(model.parameters())

# Printing the parameters of the model
for name, param in model.named_parameters():
    print(f"Parameter {name}, shape {param.shape}")

# -------------------------------------------------Dataset Class------------------------------------------------------ #
'''
The Dataset package has two most commonly used classes, 
1. data.Dataset, which provides uniform interface to access the training/test data. This class has two major functions,
   data.Dataset--|->  '__getitem__' to get ith indexed data point from the dataset 
                 |->  '__len__' to get the size of the dataset
2. data.DataLoader, which helps creates an iterator of the loaded training/test data
'''


class XORDataset(data.Dataset):
    def __init__(self, size, std=0.1):
        """
        :param size: Number of data points to be generated
        :param std: Standard deviation of the noise
        :return instantiates the XORDataset
        """
        super().__init__()
        self.size = size
        self.std = std
        self.generate_continous_xor()

    def generate_continous_xor(self):
        """
        Each data point in the generated dataset will consist of two variables, x and y. These variables are binary, i.e.,
        can take values 0 and 1, and the output is 1 if both variables have equal values.
        This dataset also illustrates benefits of using Neural Networks instead of a linear regression approach even for
        simple problems
        :return:
        """
        """Generating random Integers
        torch.randint(low=0, high, size, dtype=None, layout=torch.strided, device=None, requires_grad = False)
        :param low = inclusive (lowest desired integer value)
        :param high exlusive range (one above the desired value)
        :param size: tuple defining the shape of the output tensor
        """
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (data.sum(dim=1) == 1).to(torch.long)
        data += self.std * torch.randn(data.shape)

        self.data = data
        self.label = label

    def __len__(self):
        """
        :return: size of the dataset
        """
        return self.data.shape[0]

    def __getitem__(self, index):
        """
        :param index: index of the data and the label that needs to be returned
        :return:
        """
        return self.data[index], self.label[index]


# Visualize the samples
def visualize(data, label):
    if isinstance(data, torch.Tensor):
        data = data.numpy()
        print(data.shape)
    if isinstance(label, torch.Tensor):
        label = label.numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]
    figure = go.Figure(data=go.Scatter(x=data_0[:, 0], y=data_0[:, 1], mode='markers', name='Class 0'))
    figure.add_trace(go.Scatter(x=data_1[:, 0], y=data_1[:, 1], mode='markers', name='Class 1'))
    figure.update_layout(
        title='Dataset Samples',
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        legend_title='Legend',
    )

    # Save plot as html
    figure.write_html('continous_xor_plot.html')


dataset = XORDataset(size=200)
print("Size of dataset:", len(dataset))
print("Data point 0:", dataset[0])
visualize(dataset.data, dataset.label)

# Creating DataLoader
"""
Dataloader class represents a PyTorch iterable over the Dataset class which is helpful during the training process in 
regards to Automatic batching, multi-process loading, etc. The dataloader primarily used __getitem__ function and stacks
its output as tensors stacks it over the first dimension to form a batch. 
1. dataset: dataset from which to load the data
2. batch_size(int, optional): number of samples to be stacked per batch (default=1)
3. shuffle(bool, optional): if data needs to be returned in random order (default=False)
4. sampler(Sampler class or Iterable, optional): defines the strategy to draw samples from the dataset
5. num_workers(int, optional): Number of subprocesses to use for data loading
6. pin_memory()
7. drop_last(bool, optional): drop the last incomplete batch, if the dataset isn't divisible by batch_size (default=False)
"""
train_dataset = XORDataset(size=2000)
# Add argpraser to provide batch size from the command line
train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# Validation dataset to detect overfitting issues
validation_dataset = XORDataset(size=1000)
validation_data_loader = data.DataLoader(validation_dataset, batch_size=128, shuffle=True)

# -------------------------------------Objective Function & Optimization Method--------------------------------------- #
# Objective or Loss Function: For Binary classification we commonly use Binary Cross Entropy(BCE)
"""
Two common types of Binary loss functions
1. BCELoss()
2. BCEwithLogitLoss(): Numerically more stable
"""
loss_function = nn.BCEWithLogitsLoss()

# Optimization Method
"""
Commonly used optimization methods include, 
1. Stochastic Gradient Descent (SGD): torch.optim.SGD(params, lr=0.01, momentum=0, dampening=0. weight_decay=0, nestrov=False)
2. Adaptive Moment Estimation (Adam): torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
3. Root Mean Square Propagation (RMSprop): torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
4. Adaptive Gradient Descent (AdaGrad): torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
5. Adam Weight Decay (AdamW): torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
6. Limited-memory Broyden-Fletcher-Goldfarb-Shanno (LBFGS): 
        torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, 
        tolerance_change=1e-09, history_size=100, line_search_fn=None)
"""

# Later provide learning rate through a YAML file or through command line
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

"""
Optimizer class has two important functions
1. optimizer.step(): updates the parameters based on the gradients estimated
2. optimizer.zero_grad(): sets the gradient of all parameters as zero, used before estimating new set of gradients before
                          backpropagation
"""

# ------------------------------------------------------Training------------------------------------------------------ #
"""
Steps for Training
1. Load a batch
2. Obtain the predictions
3. Calculate the loss 
4. Backpropagation
5. Update the parameters

Additional steps for Logging
1. Create a TensorBoard Logger
2. Visualize the first computational graph (for the first batch)
3. Estimate running average of the loss
"""


def train_model(model, optimizer, data_loader, loss_module, validation_dataset, num_epochs=100, logging_dir='runs/experiments'):
    # Create TensorBoard Logger
    writer = SummaryWriter()
    model_plotted = False

    # Set model to training
    model.train()

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for input_data, input_labels in data_loader:
            # Computational graph visualized in TensorBoard
            if not model_plotted:
                writer.add_graph(model, input_data)
                model_plotted = True

            # Obtain the predictions
            predictions = model(input_data)
            predictions = predictions.squeeze(dim=1)

            # Calculate the value of loss function
            loss = loss_module(predictions, input_labels.float())

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            # Update the parameters
            optimizer.step()

            #Estimate the running average of the loss
            epoch_loss += loss.item()

        # Take average
        epoch_loss = epoch_loss/len(data_loader)

        writer.add_scalar('Training Loss',
                          epoch_loss,
                          global_step=epoch+1)

    writer.close()



train_model(model, optimizer, train_data_loader, loss_function, validation_dataset)

# ------------------------------------------Saving the state of the model--------------------------------------------- #
# Saving the trained model such that parameters can be loaded later for evaluation
state_dict = model.state_dict()
torch.save(state_dict, "Continous_XOR.tar")
# To load the exisitng parameters use torch.load(Continous_XOR.tar)
model_new = Classifier(num_inputs=2, num_hidden=4, num_outputs=1)
model_new.load_state_dict(state_dict)

# Verifying if the parameters are same
print(f"Original parameters: {model.parameters()}")
print(f"New Model's parameters: {model_new.parameters()}")

# ----------------------------------------------------Evaluation------------------------------------------------------ #

test_dataset = XORDataset(size=500)
test_data_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)


def evaluate_model(model, data_loader):
    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        for input_data, input_labels in data_loader:
            predictions = model(input_data)
            predictions = predictions.squeeze(dim=1)
            predictions = torch.sigmoid(predictions)
            prediction_labels = (predictions >= 0.5).long()

            true_labels = (prediction_labels == input_labels).sum()
            total = input_data.shape[0]
    accuracy = true_labels / total
    print(f"Accuracy of the model: {100.0 * accuracy:4.2f}%")


evaluate_model(model, test_data_loader)
# ------------------------------------Visualizing Classification Boundaries------------------------------------------- #

@torch.no_grad()
def visualize_classification(model, data, label):
    data = data.numpy()
    label = label.numpy()

    data_0 = data[label == 0]
    data_1 = data[label == 1]

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data_0[:, 0], y=data_0[:, 1], mode='markers', name='Class 0'))
    figure.add_trace(go.Scatter(x=data_1[:, 0], y=data_1[:, 1], mode='markers', name='Class 0'))
    figure.update_layout(
        title='Dataset Samples',
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        legend_title='Legend',
    )
    x1 = torch.arange(-0.5, 1.5, step=0.01)
    x2 = torch.arange(-0.5, 1.5, step=0.01)
    xx1, xx2 = torch.meshgrid(x1, x2, indexing='ij')  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    predictions = model(model_inputs)
    predictions = torch.sigmoid(predictions)
    #output_image = (1 - predictions) * c0[None, None] + predictions * c1[None, None]  # Specifying "None" in a dimension creates a new one
    #output_image = output_image.cpu().numpy()  # Convert to numpy array. This only works for tensors on CPU, hence first push to CPU
    #figure.show()
    return None


    # Save plot as html
    figure.write_html('continous_xor_plot.html')