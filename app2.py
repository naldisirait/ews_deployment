from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

#import module from this projects
from src.utils import get_current_datetime
from models.discharge_new.model_ml1 import inference_ml1
from models.inundation_new.model_ml2 import inference_ml2
from src.data_ingesting_new import get_input_ml1
from src.post_processing import output_ml1_to_dict, output_ml2_to_dict, ensure_jsonable

app = FastAPI()
def do_prediction():
    start_run_time = get_current_datetime()
    input_size_ml2 = 24

    #1. Ingest data hujan
    path_hujan_hist_72jam = "./data/hujan_72_jam.pkl"
    input_ml1, ch_wilayah, date_list, data_information, data_name_list = get_input_ml1()

    #2. Predict debit using ML1
    debit = inference_ml1(input_ml1)
    input_ml2 = debit[-input_size_ml2:].view(1,input_size_ml2,1)

    #3. Predict inundation using ML2
    genangan = inference_ml2(input_ml2)
    end_run_time = get_current_datetime()

    #Bundle output
    dates, dict_output_ml1 = output_ml1_to_dict(dates=date_list, output_ml1=debit.tolist(), precipitation=ch_wilayah)
    dict_output_ml2 = output_ml2_to_dict(dates=dates[-input_size_ml2:],output_ml2=genangan)

    output = {"Prediction Time Start": str(start_run_time), 
            "Prediction Time Finished": str(end_run_time), 
            "Prediction Output ML1": dict_output_ml1,
            "Prediction Output ML2": dict_output_ml2}

    output = ensure_jsonable(output)

    return output

@app.post("/predict")
async def predict():
    output = do_prediction()
    return output

# Run the application using the following command:
# uvicorn app2:app --reload
