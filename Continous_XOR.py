# Required Imports
import torch

# For defining the Neural Network Architecture
import torch.nn as nn
import torch.nn.functional as F

# For defining the Dataset class
import torch.utils.data as data


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
print(model)

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

data = XORDataset(size=200)
print(len(data))
print(f"13th datapoint, {data[13]}")
