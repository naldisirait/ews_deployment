from src.data_ingesting import get_precipitation_from_big_lake

if __name__ == "__main__":
    try:
        prec = get_precipitation_from_big_lake(10)
        if prec:
            print("Successfully ingested the data")
        else:
            print("Data is not available")
    except Exception as e: # Capture and raise the original exception
        # This maintains the stack trace and the original error message
        raise RuntimeError("Failed to ingest data due to an unexpected error") from e