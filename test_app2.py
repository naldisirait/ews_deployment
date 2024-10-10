from app2 import do_prediction
from datetime import datetime, timedelta
import os
from src.data_ingesting_new import get_grided_prec_palu
if __name__ == "__main__":

    current_time = datetime(2023, 11, 20, 19, 0)
    print("Try to ingest data gsmap")
    val = get_grided_prec_palu(current_time)
    output = do_prediction()
    print("App2 berjalan dengan baik.")