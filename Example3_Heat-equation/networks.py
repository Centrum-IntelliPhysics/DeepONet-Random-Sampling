import numpy as np
import torch
from torch import nn
import math
import seaborn as sns 
sns.set()

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)

class DenseNet(nn.Module):
    """
    This is a DenseNet Class.
    -> layersizes: number of neurons in each layer. 
                    E.g. [dim_in, 32, 32, 32, dim_out]
                    where, dim_in and dim_out are network's input and output dimension respectively
    -> activation: Non-linear activations function that you want to use. E.g. nn.Sigmoid(), nn.ReLU()
    -> The method model_capacity() returns the number of layers and parameters in the network.
    """
    def __init__(self, layersizes=[2, 32, 32, 32, 1], activation=nn.Sigmoid()):
        super(DenseNet, self).__init__()
        
        self.layersizes = layersizes
        self.activation = activation
        
        self.input_dim,  self.hidden_sizes, self.output_dim = self.layersizes[0], self.layersizes[1:-1], self.layersizes[-1]
        
        self.nlayers = len(self.hidden_sizes) + 1
        self.layers = nn.ModuleList([])
        for i in range(self.nlayers):
            self.layers.append( nn.Linear(self.layersizes[i], self.layersizes[i+1]) )


    def forward(self, x):
        
        for i in range(self.nlayers-1):
            x = self.activation(self.layers[i](x))
         
        # no activation for last layer
        out = self.layers[-1](x)

        return out

    def model_capacity(self):
        """
        Prints the number of parameters in the network
        """
        number_of_learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)  


class DenseResNet(nn.Module):
    """
    This is a DenseResNet Class.
    -> dim_in: network's input dimension
    -> dim_out: network's output dimension
    -> num_resnet_blocks: number of ResNet blocks
    -> num_layers_per_block: number of layers per ResNet block
    -> num_neurons: number of neurons in each layer
    -> activation: Non-linear activations function that you want to use. E.g. nn.Sigmoid(), nn.ReLU()
    -> The method model_capacity() returns the number of layers and parameters in the network.
    """
    def __init__(self, dim_in=2, dim_out=1, num_resnet_blocks=3, 
                 num_layers_per_block=2, num_neurons=50, activation=nn.Sigmoid()):
        super(DenseResNet, self).__init__()

        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.activation = activation

        self.first = nn.Linear(dim_in, num_neurons)

        self.resblocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(num_neurons, num_neurons) 
                for _ in range(num_layers_per_block)]) 
            for _ in range(num_resnet_blocks)])

        self.last = nn.Linear(num_neurons, dim_out)

    def forward(self, x):

        x = self.activation(self.first(x))

        for i in range(self.num_resnet_blocks):
            z = self.activation(self.resblocks[i][0](x))

            for j in range(1, self.num_layers_per_block):
                z = self.activation(self.resblocks[i][j](z))

            x = z + x

        out = self.last(x)

        return out

    def model_capacity(self):
        """
        Prints the number of parameters in the network
        """
        number_of_learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)  


class ConvNet(nn.Module):
    def __init__(self, input_size, n_channels, num_filters, filter_sizes, strides, paddings, poolings, end_MLP_layersizes, activation):
        """
        Convolutional Neural Network model with customizable architecture.

        Parameters:
            input_size (int or tuple): The size of the input image. If tuple, specify (height, width).
            n_channels (int): Number of input channels.
            num_filters (list): List of integers specifying the number of filters in each convolutional layer.
            filter_sizes (list): List of integers or tuples specifying the filter sizes in each convolutional layer.
            strides (list): List of integers or tuples specifying the stride for each convolutional layer.
            paddings (list): List of integers or tuples specifying the padding for each convolutional layer.
            poolings (list): List of tuples specifying the pooling type, kernel size, and stride for each pooling layer.
            end_MLP_layersizes (list): List of integers specifying the sizes of the fully connected layers at the end of the network.
            activation (torch.nn.Module): Activation function to be applied after each convolutional layer.

        Attributes:
            model (torch.nn.Module): The sequential model representing the ConvNet.

        Methods:
            forward(x): Forward pass of the model.
            model_capacity(): Prints the number of parameters in the network.
        """
        super(ConvNet, self).__init__()
        layers = []
        in_channels = n_channels
        prev_output_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)
        
        for i in range(len(filter_sizes)):
            # Convolutional layer
            conv_layer = nn.Conv2d(in_channels, num_filters[i], kernel_size=filter_sizes[i], stride=strides[i], padding=paddings[i])
            layers.append(conv_layer)
            layers.append(activation)  
            prev_output_size = self.conv_output_size(prev_output_size, filter_sizes[i], strides[i], paddings[i])  # Calculate output size with padding
            in_channels = num_filters[i]
            
            # Pooling layer
            pool_type, pool_kernel_size, pool_stride = poolings[i]
            if pool_type == 'max':
                pool_layer = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            elif pool_type == 'avg':
                pool_layer = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            elif pool_type is None:  # Check for None directly
                continue  # Skip adding a pooling layer
            layers.append(pool_layer)
            prev_output_size = self.pool_output_size(prev_output_size, pool_kernel_size, pool_stride, pool_type)  # Calculate output size
        
        layers.append(nn.Flatten())
        flattend_size = num_filters[-1] * prev_output_size[0] * prev_output_size[1]  # Adjusted flattened size
            
        # Feedforward layer
        in_features = flattend_size
        for out_features in end_MLP_layersizes[:-1]:
            feedforward_layer = nn.Linear(in_features, out_features)
            layers.append(feedforward_layer)
            layers.append(activation)
            in_features = out_features
        feedforward_layer = nn.Linear(in_features, end_MLP_layersizes[-1]) # Last feedforward layer (no activation)
        layers.append(feedforward_layer)
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
    def model_capacity(self):
        """
        Prints the number of parameters in the network
        """
        number_of_learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)

    @staticmethod
    def conv_output_size(input_size, kernel_size, stride=1, padding=0):
        """
        Calculate the output size of a convolutional layer for a rectangular input.

        Parameters:
            input_size (tuple): Input size (height, width).
            kernel_size (int or tuple): Kernel size (height, width).
            stride (int or tuple): Stride (height, width). Default is 1.
            padding (int or tuple): Padding (height, width). Default is 0.

        Returns:
            tuple: Output size of the convolutional layer (output_height, output_width).
        """
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        if not isinstance(padding, tuple):
            padding = (padding, padding)
            
        h_out = math.floor((input_size[0] + 2 * padding[0] - kernel_size[0]) / stride[0]) + 1
        w_out = math.floor((input_size[1] + 2 * padding[1] - kernel_size[1]) / stride[1]) + 1
        return (h_out, w_out)

    @staticmethod
    def pool_output_size(input_size, kernel_size, stride=1, pool_type='avg'):
        """
        Calculate the output size of a pooling layer for a rectangular input.

        Parameters:
            input_size (tuple): Input size (height, width).
            kernel_size (int or tuple): Kernel size (height, width).
            stride (int or tuple): Stride (height, width). Default is 1.
            pool_type (str): Pool type. Default is 'avg'.
        Returns:
            tuple: Output size of the pooling layer (output_height, output_width).
        """
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        if pool_type == 'avg':
            h_out = math.floor((input_size[0] - kernel_size[0]) / stride[0]) + 1
            w_out = math.floor((input_size[1] - kernel_size[1]) / stride[1]) + 1
        if pool_type == 'max':
            h_out = math.floor((input_size[0] - kernel_size[0]) / stride[0]) + 1
            w_out = math.floor((input_size[1] - kernel_size[1]) / stride[1]) + 1
        return (h_out, w_out)

# # Example usage: Rectangular input image
# input_size = (28, 32)  # Specify input size as a tuple (height, width)
# n_channels = 1
# num_filters = [40, 60, 100, 180]
# filter_sizes = [(3, 3), (3, 3), (3, 3), (3, 3)]  # Specify filter sizes as a list of tuples
# strides = [(1, 1)] * len(num_filters)  # Specify strides as a list of tuples
# paddings = [(0, 0)] * len(num_filters)  # Specify paddings as a list of tuples
# poolings = [('avg', (2, 2), (2, 2)), (None, (None, None), (None, None)), ('max', (2, 2), (2, 2)), ('avg', (2, 2), (2, 2))]  # Pooling layer specification (type, kernel_size, stride)
# p = 10
# end_MLP_layersizes = [250, 80, p]  
# activation = nn.ReLU()
# # Initialize model
# model = ConvNet(input_size, n_channels, num_filters, filter_sizes, strides, paddings, poolings, end_MLP_layersizes, activation)
# print(model)
# # Let's check the summary
# from torchsummary import summary
# summary(model, (1, 28, 32))

# # Example usage: Square input image
# input_size = (28, 28)  # Specify input size as a tuple (height, width)
# n_channels = 1
# num_filters = [40, 60, 100, 180]
# filter_sizes = [(3, 3), (3, 3), (3, 3), (3, 3)]  # Specify filter sizes as a list of tuples
# strides = [(1, 1)] * len(num_filters)  # Specify strides as a list of tuples
# paddings = [(0, 0)] * len(num_filters)  # Specify paddings as a list of tuples
# poolings = [('avg', (2, 2), (2, 2)), (None, (None, None), (None, None)), ('max', (2, 2), (2, 2)), ('avg', (2, 2), (2, 2))]  # Pooling layer specification (type, kernel_size, stride)
# p = 10
# end_MLP_layersizes = [250, 80, p]  
# activation = nn.ReLU()
# # Initialize model
# model = ConvNet(input_size, n_channels, num_filters, filter_sizes, strides, paddings, poolings, end_MLP_layersizes, activation)
# print(model)
# # Let's check the summary
# from torchsummary import summary
# summary(model, (1, 28, 28))

# # Example usage: Square input image (easy way)
# input_size = 28
# n_channels = 1
# num_filters = [40, 60, 100, 180]
# filter_sizes = [3, 3, 3, 3]
# strides = [1]*len(num_filters)
# paddings = [0]*len(num_filters)
# poolings = [('avg', 2, 2), (None, None, None), ('max', 2, 2), ('avg', 2, 2)]  # Pooling layer specification (type, kernel_size, stride)
# p = 10
# end_MLP_layersizes = [250, 80, p]  
# activation = nn.ReLU()
# # Initialize model
# model = ConvNet(input_size, n_channels, num_filters, filter_sizes, strides, paddings, poolings, end_MLP_layersizes, activation)
# print(model)
# # Let's check the summary
# from torchsummary import summary
# summary(model, (1, 28, 28))