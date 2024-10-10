import torch
import torch.nn as nn
import torch.nn.functional as F

class RegressionCNN(nn.Module):
    def __init__(self,in_channels, num_filters):
        super(RegressionCNN, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,       # number of channels image
            out_channels=num_filters,     # Number of output channels
            kernel_size=3,       # 3x3 kernel
            stride=1,            # Stride is set to 1
            padding=1            # To maintain spatial dimensions
        )
        self.bn1 = nn.BatchNorm2d(num_filters)   # Batch Normalization
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # Average Pooling
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,           # Stride is set to 1
            padding=1           # To maintain spatial dimensions
        )
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # Average Pooling
        
        # Calculate the size after convolution and pooling
        # Input: (1, 8, 7)
        # After conv1 + pool1: (16, 4, 3)
        # After conv2 + pool2: (32, 2, 1)
        self.flattened_size = num_filters * 2 * 1  # 64
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.dropout1 = nn.Dropout(0.5)  # 50% Dropout
        
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(512, in_channels)  # Output layer for regression

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 8, 7)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        # Convolutional Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Convolutional Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 64)
        
        # Fully Connected Layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Fully Connected Layer 2
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output Layer
        x = self.fc3(x)  # Shape: (batch_size, 1)
        
        return x


        