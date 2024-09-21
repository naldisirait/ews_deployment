from src.data_processing import get_input_ml1
from src.data_ingesting import get_precipitation_from_big_lake
import time

if __name__ == "__main__":
    # Track the runtime
    start_time = time.time()
    try:
        path_config_stas_to_grid = "/opt/ews/ews_deployment/configs/configuration of stasiun to grid.json"
        path_config_grid_to_subdas = "/opt/ews/ews_deployment/configs/configuration of grid to subdas.json"

        ingested_data_name = "Stasiun"
        hours = 72
        ingested_data = get_precipitation_from_big_lake(hours)

        if ingested_data != None:
            print("Successfully ingested the data")
        else:
            raise ValueError("Cannot ingest the data")
        
        all_grided_data, input_ml1 = get_input_ml1(ingested_data,
                                                   ingested_data_name,
                                                   path_config_stas_to_grid,
                                                   path_config_grid_to_subdas)
        if input_ml1 != None:
            print("Successfully process the input data for ML1")
        else:
            print("Data is not available")
    except Exception as e: # Capture and raise the original exception
        # This maintains the stack trace and the original error message
        raise RuntimeError("Failed to get the data due to an unexpected error") from e
    
    end_time = time.time()
    print(f"Runtime: {end_time - start_time} seconds")