import torch
from modelf13 import RegressionCNN
from modelf11 import Seq2Seq

def load_model_f13(device):
    #set path for the trained model
    path_trained_f13 = "./models/discharge_new/Hasil eksperimen Ml 1 best f13.pth"
    
    #set constants
    step = 1
    lag = 13 
    forecast_size = 11
    number_of_channels = 72
    num_filters = 256
    
    #defined model
    model_f13 = RegressionCNN(in_channels=number_of_channels,num_filters=num_filters)
    
    # Load the state_dict (weights) into the model
    model_f13.load_state_dict(torch.load(path_trained_f13))
    
    # Set the model to evaluation mode if you are using it for inference
    model_f13.to(device)
    model_f13.eval()
    
    return model_f13

def load_model_f11(device):
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
    path_trained_model_f11 = "./models/discharge_new/Hasil eksperimen Ml 1 forecast 11 best.pth"
    model_f11.load_state_dict(torch.load(path_trained_model_f11))
    
    # Set the model to evaluation mode if you are using it for inference
    model_f11.to(device)
    model_f11.eval()
    
    return model_f11

def inference_ml1(precip):
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
    model_f13 = load_model_f13(device)
    model_f11 = load_model_f11(device)

    #inference model with given input precipitation
    with torch.no_grad():
        output_f13 = model_f13(precip)
        B,T = output_f13.shape
        output_f13 = output_f13.view(B,T,1) # Add a new dimension, making the tensor of shape (B, T, feature)
        output_f11 = model_f11(output_f13, target_seq_len)
    discharge = torch.cat((output_f13,output_f11),axis = 1).view(-1)[-length_discharge_to_extract:]
    return discharge