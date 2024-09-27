from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import numpy as np
import time
import pickle
import torch
from datetime import datetime

#import modul from this project
from src.data_processing import get_input_ml1, get_input_hms, convert_prec_grided_to_ch_wilayah, open_json_file
from src.data_ingesting import get_prec_from_big_lake
from src.utils import inference_model, to_tensor, get_current_datetime, open_json_file
from src.post_processing import output_ml1_to_dict, output_ml2_to_dict, ensure_jsonable

def get_input_debit_sample(name):
    try:
        with open('Kasus Validasi ML2.pkl', 'rb') as file:
            data = pickle.load(file)
        
        debit = data[name]  # Ensure 'debit' is extracted correctly
        len_flat = len(debit)

        # Make sure 'debit' is a NumPy array
        if not isinstance(debit, np.ndarray):
            debit = np.array(debit)  # Convert to a NumPy array if it's not already
        debit = debit.tolist()
        # Convert to a PyTorch tensor
        debit = torch.tensor(debit, dtype=torch.float32)
        #debit = torch.from_numpy(debit)

        # Reshape to match the required shape
        debit = debit.reshape(1, len_flat)
        debit = debit.numpy()

    except Exception as e:
        print(f"An error occurred: {e}")
        raise  # Re-raise the error after logging it
    return debit

# Define input data model
class InputData(BaseModel):
    inputs: List[List[float]]

#import model ml1 and ml2
from models.discharge.model_ml1 import load_model_ml1
from models.inundation.model_ml2  import load_model_ml2

app = FastAPI()
def do_prediction():
    tstart = time.time()
    kasus = "Kasus 8"
    dummy = False
    start_run_pred = get_current_datetime()

    #1. Define all constants and load models
    hours_hms = 720
    hours = 144
    jumlah_subdas = 114
    input_size_ml1 = hours * jumlah_subdas #jumlah jam dikali jumlah subdas
    output_size_ml1 = 168 #jumlah debit yang diestimasi, 24 jam terakhir adalah hasil forecast
    model_ml1 = load_model_ml1(input_size=input_size_ml1, output_size=output_size_ml1)

    input_size_ml2 = 72
    output_size_ml2 = 3078 * 2019 #rows x columns 
    model_ml2 = load_model_ml2(input_size=input_size_ml2, output_size=output_size_ml2)

    #2. Ingest Data input
    path_config_stas_to_grid = "/opt/ews/ews_deployment/configs/configuration of stasiun to grid.json"
    path_config_grid_to_subdas = "/opt/ews/ews_deployment/configs/configuration of grid to subdas.json"
    path_config_grid_to_df = "/opt/ews/ews_deployment/configs/configuration of grided to df.json"
    conf_grid_to_df = open_json_file(path_config_grid_to_df)
    index_grided_chosen = conf_grid_to_df['indexes']

    #ingested_data_name, ingested_data, runtime_ingest_data = get_prec_from_big_lake(hours)
    ingested_data_name_hms, ingested_data_hms,runtime_ingest_data = get_prec_from_big_lake(hours_hms)
    print(f"Ingested data runtime {runtime_ingest_data}")
    print(f"ingested_data type {type(ingested_data_hms)}")
    print(f"Ingest data satelit sukses!")

if __name__ == "__main__":
    do_prediction()
    
