from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import numpy as np
import pickle
import torch
from datetime import datetime

#import modul from this project
from src.data_processing import get_input_ml1, get_input_hms
from src.data_ingesting import get_prec_from_big_lake
from src.utils import inference_model,to_tensor
from src.post_processing import output_ml1_to_dict, output_ml2_to_dict

def get_input_debit_sample(name):
    try:
        with open('Kasus Validasi ML2.pkl', 'rb') as file:
            data = pickle.load(file)
        
        debit = data[name]  # Ensure 'debit' is extracted correctly
        len_flat = len(debit)

        # Make sure 'debit' is a NumPy array
        if not isinstance(debit, np.ndarray):
            debit = np.array(debit)  # Convert to a NumPy array if it's not already

        # Convert to a PyTorch tensor
        debit = torch.tensor(debit, dtype=torch.float32)
        #debit = torch.from_numpy(debit)

        # Reshape to match the required shape
        debit = debit.reshape(1, len_flat)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise  # Re-raise the error after logging it
    return debit

def get_current_datetime():
    # Get the current date and time
    now = datetime.now()
    # Format the date and time as YYYY-MM-DD HH:MM:SS
    return now.strftime("%Y-%m-%d %H:%M:%S")

# Define input data model
class InputData(BaseModel):
    inputs: List[List[float]]

#import model ml1 and ml2
from models.discharge.model_ml1 import load_model_ml1
from models.inundation.model_ml2  import load_model_ml2

app = FastAPI()
def do_prediction():
    cwd = os.getcwd()
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
    #path_config_grid_to_df = "/opt/ews/ews_deployment/configs/configuration of grided to df.json"

    ingested_data_name, ingested_data = get_prec_from_big_lake(hours)
    #ingested_data_name_hms, ingested_data_hms = get_prec_from_big_lake(hours_hms)

    #3.1 Inference ML1
    all_grided_data, dates, input_ml1 =  get_input_ml1(ingested_data,
                                                   ingested_data_name,
                                                   path_config_stas_to_grid,
                                                   path_config_grid_to_subdas)
    
    all_grided_data = np.array(all_grided_data)
    output_ml1 = inference_model(model_ml1,input_ml1)

    #3.2 Prepare data hms and inference HEC-HMS
    # all_grided_data_hms, df_hms = get_input_hms(ingested_data=ingested_data_hms,
    #                                             ingested_data_name=ingested_data_name_hms, 
    #                                             path_conf_grided_to_df=path_config_grid_to_df,
    #                                             path_config_stas_to_grid=path_config_stas_to_grid,
    #                                             path_config_grid_to_subdas=path_config_grid_to_subdas)
    
    #4.1 Inference ML2 using output from ML1
    output_ml1 = output_ml1[:,-input_size_ml2:]
    kasus = "Kasus 8"
    dummy = False
    if dummy:
        output_ml1 = get_input_debit_sample(kasus)
        
    input_ml2 = np.expand_dims(output_ml1, axis=-1)
    input_ml2 = to_tensor(input_ml2)
    input_ml2 = torch.tensor(input_ml2, dtype=torch.float32)
    output_ml2 = inference_model(model_ml2, input_ml2)
    output_ml2 = output_ml2[0,:].reshape(3078,2019)
    print(output_ml2.shape)

    #4.2 Inference ML2 using output from HMS
    #5. Bundle the Output
    #Convert output ml1 to dict
    dates, dict_output_ml1 = output_ml1_to_dict(dates=dates, output_ml1=output_ml1[0,:].tolist(), precipitation=all_grided_data.tolist())
    #Convert output ml2 to dict
    dict_output_ml2 = output_ml2_to_dict(dates=dates[-input_size_ml2:],output_ml2=output_ml2)
    
    end_run_pred = get_current_datetime()
    return start_run_pred, end_run_pred, dict_output_ml1, dict_output_ml2

@app.post("/predict")
async def predict():
    start_run_pred, end_run_pred, dict_output_ml1, dict_output_ml2 = do_prediction()
    #return {"Output": outputs.tolist(),"Prediction Time": start_run_pred, "Prediction time Finished": end_run_pred}
    output = {"Prediction Time Start": str(start_run_pred), 
              "Prediction time Finished": str(end_run_pred), 
              "Prediction Output ml1": dict_output_ml1,
              "Prediction Output ml2": dict_output_ml2}

    # output = {"Prediction Time Start": str(start_run_pred), 
    #         "Prediction time Finished": str(end_run_pred), 
    #         "Prediction Output ml1": dict_output_ml1}
    
    return output

#Local test
# uvicorn app:app --reload
