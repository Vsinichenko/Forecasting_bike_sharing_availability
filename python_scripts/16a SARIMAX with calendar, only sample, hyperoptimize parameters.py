#!/usr/bin/env python
# coding: utf-8

import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime

import numpy as np

# import pmdarima.arima as pm_arima
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.simplefilter(action="ignore", category=FutureWarning)

mycell = "871f1b559ffffff"

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
for df_tmp in [df_DD, df_FB]:
    df_tmp["weekday"] = df_tmp.datetime_hour.dt.dayofweek
    df_tmp["weekday"] = df_tmp["weekday"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
    weekday_df = pd.get_dummies(df_tmp["weekday"], prefix="weekday", drop_first=False, dtype=int)
    weekday_df.index = df_tmp.index
    weekday_df.drop(columns="weekday_Mon", inplace=True)
    df_tmp[weekday_df.columns] = weekday_df

    df_tmp["hour"] = df_tmp.datetime_hour.dt.hour
    hours_df = pd.get_dummies(df_tmp["hour"], prefix="hour", drop_first=False, dtype=int)
    hours_df.index = df_tmp.index
    hours_df.drop(columns="hour_0", inplace=True)
    df_tmp[hours_df.columns] = hours_df
    df_tmp["is_dayoff"] = df_tmp["weekday_Sat"] + df_tmp["weekday_Sun"]
    # list of german holidays in 2023 and 2024
    german_holidays = ["2024-01-01", "2024-03-20", "2024-09-20", "2024-10-03", "2023-06-08"]
    german_holidays = [pd.to_datetime(date).date() for date in german_holidays]
    flt = df_tmp.datetime_hour.dt.date.isin(german_holidays)
    len(df_tmp[flt])
    df_tmp.loc[flt, "is_dayoff"] = 1


df_DD_1 = df_DD.loc[df_DD.datetime_hour.dt.date <= test_range_1_DD[-1]]  # first half of dates
df_DD_2 = df_DD.loc[df_DD.datetime_hour.dt.date > test_range_1_DD[-1]]  # second half of dates

flt = df_DD_1.datetime_hour.dt.date.isin(test_range_1_DD)
train_validation_DD_1 = df_DD_1.loc[~flt]
test_DD_1 = df_DD_1.loc[flt].sort_values("datetime_hour")

flt = df_DD_2.datetime_hour.dt.date.isin(test_range_2_DD)
train_validation_DD_2 = df_DD_2.loc[~flt].sort_values("datetime_hour")
test_DD_2 = df_DD_2.loc[flt].sort_values("datetime_hour")

# FB
df_FB_1 = df_FB.loc[df_FB.datetime_hour.dt.date <= test_range_1_FB[-1]]
df_FB_2 = df_FB.loc[df_FB.datetime_hour.dt.date > test_range_1_FB[-1]]

flt = df_FB_1.datetime_hour.dt.date.isin(test_range_1_FB)
train_validation_FB_1 = df_FB_1.loc[~flt]
test_FB_1 = df_FB_1.loc[flt].sort_values("datetime_hour")

flt = df_FB_2.datetime_hour.dt.date.isin(test_range_2_FB)
train_validation_FB_2 = df_FB_2.loc[~flt].sort_values("datetime_hour")
test_FB_2 = df_FB_2.loc[flt].sort_values("datetime_hour")


df_helper = {"DD": df_DD, "FB": df_FB}
dep_var_helper = {"demand": "rent_count", "supply": "return_count"}
train_df_helper = {"DD": {1: train_validation_DD_1, 2: train_validation_DD_2}, "FB": {1: train_validation_FB_1, 2: train_validation_FB_2}}
test_df_helper = {"DD": {1: test_DD_1, 2: test_DD_2}, "FB": {1: test_FB_1, 2: test_FB_2}}

for city in ["DD"]:
    for current_cell in [mycell]:
        for part in [1]:
            for dep_var in ["demand"]:
                model_name = f"sarimax_{city}_{dep_var}_part_{part}_cell_{current_cell}.pkl"
                model_dir = "models/sarimax"
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

                model_path = os.path.join(model_dir, model_name)

                # if os.path.exists(model_path):
                #     continue

                logging.info(f"CITY {city} CURRENT CELL {current_cell}, PART {part}, DEPVAR {dep_var}")
                dep_colname = dep_var_helper[dep_var]
                train_df = train_df_helper[city][part]
                train_df = train_df.loc[train_df.hex_id == current_cell].set_index("datetime_hour")

                test_df = test_df_helper[city][part]
                test_df = test_df.loc[test_df.hex_id == current_cell].set_index("datetime_hour")

                train_sr = train_df[dep_colname]
                exog_colnames = [
                    "hour_1",
                    "hour_2",
                    "hour_3",
                    "hour_4",
                    "hour_5",
                    "hour_6",
                    "hour_7",
                    "hour_8",
                    "hour_9",
                    "hour_10",
                    "hour_11",
                    "hour_12",
                    "hour_13",
                    "hour_14",
                    "hour_15",
                    "hour_16",
                    "hour_17",
                    "hour_18",
                    "hour_19",
                    "hour_20",
                    "hour_21",
                    "hour_22",
                    "hour_23",
                    "weekday_Tue",
                    "weekday_Wed",
                    "weekday_Thu",
                    "weekday_Fri",
                    "weekday_Sat",
                    "weekday_Sun",
                    "is_dayoff",
                ]

                train_exog_df = train_df[exog_colnames]

                test_sr = test_df[dep_colname]
                test_exog_df = test_df[exog_colnames]

                if city == "FB" and part == 1:
                    train_sr = train_sr.asfreq("h", fill_value=train_sr.mean())
                else:
                    train_sr = train_sr.asfreq("h")

                start_train_time = time.time()

                tasks = [(p, q, P, Q) for p in [0, 1] for q in [0, 1] for P in [0, 1] for Q in [0, 1]]
                for p, q, P, Q in tasks:
                    logging.info(f"Trying p={p}, d=1, q={q}, P={P}, D=0, Q={Q}, s=24")
                    try:
                        model = SARIMAX(endog=train_sr, exog=train_exog_df, order=(p, 1, q), seasonal_order=(P, 0, Q, 24), freq="h")
                        results = model.fit()
                        # print model summary

                        logging.info(results.summary())

                        logging.info(f"Elapsed time: {(time.time() - start_train_time)/60} minutes")
                        predictions = results.get_forecast(steps=len(test_sr), exog=test_exog_df).predicted_mean
                        x = test_sr.index
                        plt.plot(x, test_sr, color="black")
                        plt.title(model_name)
                        plt.scatter(x, predictions, color="yellow")
                        plt.savefig(f"tmp/{time.time()}_{p}_{q}_{P}_{Q}.png")
                        plt.close()

                        rmse = np.sqrt(mean_squared_error(test_sr, predictions))
                        logging.info(f"RMSE: {rmse} for p={p}, d=1, q={q}, P={P}, D=0, Q={Q}, s=24")

                    except Exception as e:
                        logging.error(f"Error with p={p}, d=1, q={q}, P={P}, D=0, Q={Q}, s=24: {e}")
                        continue

                # with open(model_path, "wb") as pkl:
                #     pickle.dump(model, pkl)
                # logging.info(f"Model saved as {model_name}")
