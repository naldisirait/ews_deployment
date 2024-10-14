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

# Seq2Seq Model Components: Encoder, Decoder, Seq2Seq
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.3):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        outputs, (hidden, cell) = self.lstm(x, (h0, c0))
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, dropout=0.3):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout=0.3):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(output_dim, hidden_dim, num_layers, dropout)

    def forward(self, src, target_len):
        batch_size = src.size(0)
        input_dim = src.size(2)
        hidden, cell = self.encoder(src)
        decoder_input = torch.zeros(batch_size, 1, input_dim).to(src.device)
        predictions = torch.zeros(batch_size, target_len, input_dim).to(src.device)
        for t in range(target_len):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            predictions[:, t:t+1, :] = prediction
            decoder_input = prediction
        return predictions

def load_model_f13(device,path_trained_f13):
    #set constants
    step = 1
    lag = 13 
    forecast_size = 11
    number_of_channels = 72
    num_filters = 256
    
    #defined model
    model_f13 = RegressionCNN(in_channels=number_of_channels,num_filters=num_filters)
    
    # Load the state_dict (weights) into the model
    model_f13.load_state_dict(torch.load(path_trained_f13,map_location=torch.device('cpu')))
    
    # Set the model to evaluation mode if you are using it for inference
    model_f13.to(device)
    model_f13.eval()
    
    return model_f13

def load_model_f11(device,path_trained_f11):
    # Hyperparameters
    input_dim = 1        # Single feature for univariate time series
    output_dim = 1       # Predict one value per time step
    hidden_dim = 64      # Hidden state size in LSTM
    num_layers = 2      # Number of LSTM layers
    dropout = 0.5      # Dropout rate
    input_seq_len = 72 # Length of input sequence (lookback)
    target_seq_len = 11  # Length of target sequence (prediction steps)
    
    # Model instantiation
    model_f11 = Seq2Seq(input_dim, output_dim, hidden_dim, num_layers, dropout).to(device)
    
    #load model
    model_f11.load_state_dict(torch.load(path_trained_f11, map_location=torch.device('cpu')))
    
    # Set the model to evaluation mode if you are using it for inference
    model_f11.to(device)
    model_f11.eval()
    
    return model_f11

def inference_ml1(precip, config):
    """
    Function to predict discharge
    Args:
        precip(tensor): grided precipitation with shape (Batch=1, len_history=72, width=8, height=7)
    Returns:
        discharge(tensor): 72 hours of discharge, where 48 hours is estimated and 24 hours forcast discharge.
    """
    #set constants
    device = "cpu"
    target_seq_len = 11
    length_discharge_to_extract = 72
    
    #load model
    path_trained_f13 = config['model']['path_trained_ml1_f13']
    path_trained_f11 = config['model']['path_trained_ml1_f11']
    model_f13 = load_model_f13(device,path_trained_f13=path_trained_f13)
    model_f11 = load_model_f11(device,path_trained_f11=path_trained_f11)

    #inference model with given input precipitation
    with torch.no_grad():
        output_f13 = model_f13(precip)
        B,T = output_f13.shape
        output_f13 = output_f13.view(B,T,1) # Add a new dimension, making the tensor of shape (B, T, feature)
        output_f11 = model_f11(output_f13, target_seq_len)
    discharge = torch.cat((output_f13,output_f11),axis = 1).view(-1)[-length_discharge_to_extract:]
    return discharge