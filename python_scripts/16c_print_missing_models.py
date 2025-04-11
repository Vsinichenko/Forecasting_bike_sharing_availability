#!/usr/bin/env python
# coding: utf-8


import logging
import os
import sys
import warnings
from datetime import datetime

# import pmdarima.arima as pm_arima
import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)


mycell = "871f1b559ffffff"

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = "logs/sarimax_calendar"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_fullpath = os.path.join(log_dir, f"sarimax_calendar_{start_time}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_fullpath), logging.StreamHandler(sys.stdout)],
)


class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self.buffer = ""

    def write(self, message):
        if message.strip():  # Avoid empty messages
            self.level(message.strip())

    def flush(self):
        pass  # No need to flush manually


sys.stdout = LoggerWriter(logging.info)
sys.stderr = LoggerWriter(logging.error)  # Capture warnings and errors


logging.info("Reading data")

file_datetime = "2025-03-19_10-47-56"
filename_DD = f"data/nextbike/hourly_demand_supply_Dresden {file_datetime}.csv"
filename_FB = f"data/nextbike/hourly_demand_supply_Freiburg {file_datetime}.csv"
df_DD = pd.read_csv(filename_DD, index_col=None, parse_dates=["datetime_hour"])
df_FB = pd.read_csv(filename_FB, index_col=None, parse_dates=["datetime_hour"])


df_helper = {"DD": df_DD, "FB": df_FB}


model_dir = "models/sarimax_calendar"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
img_dir = "tmp/sarimax_calendar"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)


for city in ["DD", "FB"]:
    for current_cell in df_helper[city].hex_id.unique():
        for part in [1, 2]:
            for dep_var in ["demand", "supply"]:
                model_name = f"sarimax_calendar_{city}_{dep_var}_part_{part}_cell_{current_cell}.pkl"
                model_path = os.path.join(model_dir, model_name)
                if not os.path.exists(model_path):
                    logging.info(f"Model {model_name} is missing")
