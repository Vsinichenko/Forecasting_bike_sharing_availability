#!/usr/bin/env python
# coding: utf-8

import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
from pmdarima import auto_arima

warnings.simplefilter(action="ignore", category=FutureWarning)

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_fullpath = f"logs/all_hexagons_arima_{start_time}.log"

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

# test date ranges
test_range_1_DD = pd.date_range(start="2024-03-21", end="2024-03-31")
test_range_1_DD = [date.date() for date in test_range_1_DD]

test_range_2_DD = pd.date_range(start="2024-10-21", end="2024-10-31")
test_range_2_DD = [date.date() for date in test_range_2_DD]

test_range_1_FB = pd.date_range(start="2023-07-24", end="2023-07-31")
test_range_1_FB = [date.date() for date in test_range_1_FB]

test_range_2_FB = pd.date_range(start="2024-10-23", end="2024-10-31")
test_range_2_FB = [date.date() for date in test_range_2_FB]

## slice dataframes
# DD
df_DD_1 = df_DD.loc[df_DD.datetime_hour.dt.date <= test_range_1_DD[-1]]
df_DD_2 = df_DD.loc[df_DD.datetime_hour.dt.date > test_range_1_DD[-1]]

flt = df_DD_1.datetime_hour.dt.date.isin(test_range_1_DD)
train_validation_DD_1 = df_DD_1.loc[~flt]
test_DD_1 = df_DD.loc[flt].sort_values("datetime_hour")

flt = df_DD_2.datetime_hour.dt.date.isin(test_range_2_DD)
train_validation_DD_2 = df_DD_2.loc[~flt].sort_values("datetime_hour")
test_DD_2 = df_DD_2.loc[flt].sort_values("datetime_hour")

# FB
df_FB_1 = df_FB.loc[df_FB.datetime_hour.dt.date <= test_range_1_FB[-1]]
df_FB_2 = df_FB.loc[df_FB.datetime_hour.dt.date > test_range_1_FB[-1]]

flt = df_FB_1.datetime_hour.dt.date.isin(test_range_1_FB)
train_validation_FB_1 = df_FB_1.loc[~flt]
test_FB_1 = df_FB.loc[flt].sort_values("datetime_hour")

flt = df_FB_2.datetime_hour.dt.date.isin(test_range_2_FB)
train_validation_FB_2 = df_FB_2.loc[~flt].sort_values("datetime_hour")
test_FB_2 = df_FB_2.loc[flt].sort_values("datetime_hour")


df_helper = {"DD": df_DD, "FB": df_FB}
dep_var_helper = {"demand": "rent_count", "supply": "return_count"}
train_df_helper = {"DD": {1: train_validation_DD_1, 2: train_validation_DD_2}, "FB": {1: train_validation_FB_1, 2: train_validation_FB_2}}
test_df_helper = {"DD": {1: test_DD_1, 2: test_DD_2}, "FB": {1: test_FB_1, 2: test_FB_2}}

for city in ["DD"]:
    for current_cell in df_helper[city].hex_id.unique():
        for part in [1, 2]:
            for dep_var in ["demand", "supply"]:
                model_name = f"models/sarima_{city}_{dep_var}_part_{part}_cell_{current_cell}.pkl"
                if os.path.exists(model_name):
                    continue

                logging.info(f"CITY {city} CURRENT CELL {current_cell}, PART {part}, DEPVAR {dep_var}")
                dep_colname = dep_var_helper[dep_var]
                train_df = train_df_helper[city][part]
                test_df = test_df_helper[city][part]
                train = train_df[train_df.hex_id == current_cell].set_index("datetime_hour")[dep_colname]
                test = test_df[test_df.hex_id == current_cell].set_index("datetime_hour")[dep_colname]

                train = train.asfreq("h")
                test = test.asfreq("h")

                start_train_time = time.time()
                logging.info("Start ARIMA optimisation")

                # max_p=3, max_d=2, max_q=3,
                model = auto_arima(y=train, trace=True, stepwise=True, suppress_warnings=False, seasonal=True, m=24, n_jobs=-1)

                model.fit(train)

                logging.info(f"Elapsed time: {(time.time() - start_train_time)/60} minutes")

                logging.info(model.summary())

                predictions = model.predict(n_periods=len(test))

                x = test.index
                plt.plot(x, test, color="black")
                plt.title(model_name)
                plt.scatter(x, predictions, color="yellow")
                plt.savefig(f"tmp/{model_name}.png")
                plt.close()

                with open(model_name, "wb") as pkl:
                    pickle.dump(model, pkl)
                logging.info(f"Model saved as {model_name}")
