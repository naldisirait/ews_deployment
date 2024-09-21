from src.data_processing import get_input_ml1

if __name__ == "__main__":
    try:
        data = get_input_ml1(10)
        if data != None:
            print("Successfully process the input data for ML1")
        else:
            print("Data is not available")
    except Exception as e: # Capture and raise the original exception
        # This maintains the stack trace and the original error message
        raise RuntimeError("Failed to get the data due to an unexpected error") from e