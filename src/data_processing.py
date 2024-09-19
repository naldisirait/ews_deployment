import numpy as np
import torch

def data_prep_for_inference_fc(input_data):
    #convert data into array
    inputs = np.array(input_data)
    
    #sum all grid data
    inputs = np.sum(inputs, axis = (1,2))
    inputs = np.expand_dims(inputs, axis = 0)
    
    #convert array to tensor
    inputs = torch.tensor(inputs, dtype=torch.float32)

    return inputs

def create_lstm_dataset(data, input_sequence_len, target_sequence_len):
    """
    Prepare time series data for LSTM model with variable input and target sequence lengths.

    Parameters:
    data (np.ndarray or list): The time series data.
    input_sequence_len (int): The number of time steps in each input sequence.
    target_sequence_len (int): The number of time steps in each target sequence.

    Returns:
    np.ndarray: Array containing the input sequences.
    np.ndarray: Array containing the target sequences.
    """
    data = np.array(data)
    X, y = [], []
    for i in range(len(data) - input_sequence_len - target_sequence_len + 1):
        X.append(data[i:i + input_sequence_len])
        y.append(data[i + input_sequence_len:i + input_sequence_len + target_sequence_len])

    X,y = np.array(X), np.array(y)
    # Convert NumPy array to PyTorch tensor
    X = torch.tensor(X, dtype=torch.float32)
    X = X.unsqueeze(-1)  # Adds a dimension at the last position
    
    # Convert NumPy array to PyTorch tensor
    y = torch.tensor(y, dtype=torch.float32)
    y = y.unsqueeze(-1)  # Adds a dimension at the last position

    return X,y

def seperate_train_test_sequence(data, train_size):
    """
    Parameters:
    data: array like, time series data
    train_size: float, the size of training data

    Returns:
    data_train: array like, time series data for training
    data_test: array like, time series data for testing
    """
    #get thhe index of the last training data
    idx_train_end = int(len(data) * train_size)

    #slice the data
    data_train = data[0:idx_train_end]
    data_test = data[idx_train_end:]
    
    return data_train, data_test