import pickle

def load_pickle(file_path):
    """
    Load data from a pickle file.

    Parameters:
    file_path (str): The path to the pickle file.

    Returns:
    object: The data loaded from the pickle file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data