from app import do_prediction
from datetime import datetime, timedelta
import os
from src.data_ingesting import get_grided_prec_palu
if __name__ == "__main__":
    current_time = datetime(2024, 8, 20, 19, 0)
    print("Try to ingest data gsmap")
    val = get_grided_prec_palu(current_time)
    output = do_prediction()
    print("App berjalan dengan baik.")