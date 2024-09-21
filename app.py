from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from typing import List
import numpy as np
from datetime import datetime
import re

#import modul from this projects
from models.discharge.FullyConnectedModel import FullyConnectedModel

from src.gsmap_data_pipeline import get_input_precip_data
from src.data_processing import get_input_ml1
from src.data_ingesting import get_precipitation_from_big_lake
from src.utils import inference_model,to_tensor
from src.post_processing import convert_array_to_tif, output_ml1_to_json, output_ml1_to_dict, output_ml2_to_dict

def get_current_datetime():
    # Get the current date and time
    now = datetime.now()
    # Format the date and time as YYYY-MM-DD HH:MM:SS
    return now.strftime("%Y-%m-%d %H:%M")

# Define input data model
class InputData(BaseModel):
    inputs: List[List[float]]

#import model ml1 and ml2
from models.discharge.model_ml1 import load_model_ml1
from models.inundation.model_ml2  import load_model_ml2

app = FastAPI()
def do_prediction():
    start_run_pred = get_current_datetime()

    hours = 144
    jumlah_subdas = 114
    input_size_ml1 = hours * jumlah_subdas #jumlah jam dikali jumlah subdas
    output_size_ml1 = 144 #jumlah debit yang diestimasi, 24 jam terakhir adalah hasil forecast
    model_ml1 = load_model_ml1(input_size=input_size_ml1, output_size=output_size_ml1)

    input_size_ml2 = 72
    output_size_ml2 = 3078 * 2019 #rows x columns 
    model_ml2 = load_model_ml2(input_size=input_size_ml2, output_size=output_size_ml2)

    #input_ml1 = to_tensor(np.random.rand(1,16416))
    path_config_stas_to_grid = "/opt/ews/ews_deployment/configs/configuration of stasiun to grid.json"
    path_config_grid_to_subdas = "/opt/ews/ews_deployment/configs/configuration of grid to subdas.json"

    ingested_data_name = "Stasiun"
    ingested_data = get_precipitation_from_big_lake(hours)

    all_grided_data, dates, input_ml1 =  get_input_ml1(ingested_data,
                                                   ingested_data_name,
                                                   path_config_stas_to_grid,
                                                   path_config_grid_to_subdas)

    output_ml1 = inference_model(model_ml1,input_ml1)
    print(output_ml1.shape)

    # #Convert output ml1 to dict
    dict_output_ml1 = output_ml1_to_dict(dates=dates, output_ml1=output_ml1[0,:].tolist())
    
    output_ml1 = output_ml1[:,-input_size_ml2:]
    input_ml2 = np.expand_dims(output_ml1, axis=-1)
    input_ml2 = to_tensor(input_ml2)
    output_ml2 = inference_model(model_ml2, input_ml2)
    print(output_ml2.shape)

    #Convert output ml2 to dict
    dict_output_ml2 = output_ml2_to_dict(dates=dates[-input_size_ml2:],output_ml2=output_ml2)
    # #convert the array to tiff
    # filenametif = f"Prediction flood {start_run_pred}"
    # filenametif = re.sub(r'[:,.-]', '', filenametif)
    # convert_array_to_tif(data_array=output_ml2, filename=f"{filenametif}.tif")
    
    end_run_pred = get_current_datetime()
    #timestamp_end = end_run_pred.strftime('%Y-%m-%d_%H-%M-%S')
    return start_run_pred, end_run_pred, dict_output_ml1, dict_output_ml2

@app.post("/predict")
async def predict():
    start_run_pred, end_run_pred, dict_output_ml1, dict_output_ml2 = do_prediction()
    #return {"Output": outputs.tolist(),"Prediction Time": start_run_pred, "Prediction time Finished": end_run_pred}
    # output = {"Prediction Time Start": str(start_run_pred), 
    #           "Prediction time Finished": str(end_run_pred), 
    #           "Prediction Output ml1": dict_output_ml1,
    #           "Prediction Output ml2": dict_output_ml2}

    output = {"Prediction Time Start": str(start_run_pred), 
            "Prediction time Finished": str(end_run_pred), 
            "Prediction Output ml1": dict_output_ml1}
    
    return output

# Run the application using the following command:
# uvicorn app:app --reload
