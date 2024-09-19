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

from src.gsmap_data_pipeline import get_data_gsmap_now, get_input_precip_data
from src.data_processing import data_prep_for_inference_fc
from src.utils import inference_model,to_tensor
from src.post_processing import convert_array_to_tif, output_ml1_to_json

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

input_size_ml1 = 144 * 114 #jumlah jam dikali jumlah hari
output_size_ml1 = 144 #jumlah debit yang diestimasi, 24 jam terakhir adalah hasil forecast
model_ml1 = load_model_ml1(input_size=input_size_ml1, output_size=output_size_ml1)

input_size_ml2 = 72
output_size_ml2 = 3078 * 2019 #rows x columns 
model_ml2 = load_model_ml2(input_size=input_size_ml2, output_size=output_size_ml2)

# Define a prediction endpoint
# async def predict(data: InputData):
#     start_run_pred = get_current_datetime()
#     inputs = data_prep_for_inference_fc(data.inputs)

#     print(inputs.shape)


app = FastAPI()
def do_prediction():
    start_run_pred = get_current_datetime()
    #timestamp_start = start_run_pred.strftime('%Y-%m-%d_%H-%M-%S')
    input_ml1 = to_tensor(np.random.rand(1,16416))
    output_ml1 = inference_model(model_ml1,input_ml1)

    #Save to json file
    filename_json = f"Predicted Debit {start_run_pred}"
    filename_json = re.sub(r'[:,.-]', '', filename_json)
    output_ml1_to_json(values=output_ml1, filename=f"{filename_json}.json",prediction_time=start_run_pred)
    print(output_ml1.shape)
    output_ml1 = output_ml1[:,-72:]
    input_ml2 = np.expand_dims(output_ml1, axis=-1)
    input_ml2 = to_tensor(input_ml2)
    output_ml2 = inference_model(model_ml2, input_ml2)
    print(output_ml2.shape)
    
    #convert the array to tiff
    filenametif = f"Prediction flood {start_run_pred}"
    filenametif = re.sub(r'[:,.-]', '', filenametif)
    convert_array_to_tif(data_array=output_ml2, filename=f"{filenametif}.tif")
    
    
    end_run_pred = get_current_datetime()
    #timestamp_end = end_run_pred.strftime('%Y-%m-%d_%H-%M-%S')
    return start_run_pred, end_run_pred

@app.post("/predict")
async def predict():
    start_run_pred, end_run_pred = do_prediction()
    #return {"Output": outputs.tolist(),"Prediction Time": start_run_pred, "Prediction time Finished": end_run_pred}
    return {"Prediction Time Start": str(start_run_pred), "Prediction time Finished": str(end_run_pred)}

#Define data pipeline endpoint
@app.post("/get_input_data")
async def get_input_data():
    #download latest data
    get_data_gsmap_now(path_data="data_gsmap_now")

    #get precipitation value for input modelling
    precip_val = get_input_precip_data(path_data="data_gsmap_now/gsmap_now_palu", days=3)

    prec_dict = {"Precip Value": precip_val.tolist()}
    
    return prec_dict

# Run the application using the following command:
# uvicorn app:app --reload
