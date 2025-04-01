#!/usr/bin/env python
# coding: utf-8

from statsmodels.tsa.statespace.sarimax import SARIMAX

# import pmdarima.arima as pm_arima
import pandas as pd
import time
import pickle
import logging
import sys
from datetime import datetime
from matplotlib import pyplot as plt
import warnings
import os

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
                model_name = f"sarima_{city}_{dep_var}_part_{part}_cell_{current_cell}.pkl"
                model_dir = "models/sarima_fixed"
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

                model_path = os.path.join(model_dir, model_name)

                # if os.path.exists(model_path):
                #     continue

                logging.info(f"CITY {city} CURRENT CELL {current_cell}, PART {part}, DEPVAR {dep_var}")
                dep_colname = dep_var_helper[dep_var]
                train_df = train_df_helper[city][part]
                test_df = test_df_helper[city][part]
                train = train_df[train_df.hex_id == current_cell].set_index("datetime_hour")[dep_colname]
                test = test_df[test_df.hex_id == current_cell].set_index("datetime_hour")[dep_colname]
                start_train_time = time.time()

                # one boolean column for each day of the week
                weekday = train.index.weekday
                # turn weekday integers into words
                weekday = weekday.map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
                # boolean into 1-0
                weekday = pd.get_dummies(weekday, prefix="weekday", drop_first=False, dtype=int)
                weekday.drop(columns="weekday_Mon", inplace=True)
                weekday.index = train.index

                # boolean into 1-0
                # hour = train.index.hour

                # # exog_df = pd.concat([pd.Series(is_weekend, name="is_weekend"), pd.Series(hour, name="hour")], axis=1)
                # exog_df.index = train.index

                model = SARIMAX(endog=train, exog=weekday, order=(1, 1, 0), seasonal_order=(1, 0, 0, 24), freq="h")

                results = model.fit()

                logging.info(f"Elapsed time: {(time.time() - start_train_time)/60} minutes")

                logging.info(results.summary())

                weekday = test.index.weekday
                # turn weekday integers into words
                weekday = weekday.map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
                # boolean into 1-0
                weekday = pd.get_dummies(weekday, prefix="weekday", drop_first=False, dtype=int)
                weekday.drop(columns="weekday_Mon", inplace=True)
                weekday.index = test.index

                # exog_df_test = pd.concat([pd.Series(is_weekend_test, name="is_weeked"), pd.Series(hour_test, name="hour")], axis=1)

                predictions = results.get_forecast(steps=len(test), exog=weekday).predicted_mean
                print("Predictions:")
                print(predictions)

                x = test.index
                plt.plot(x, test, color="black")
                plt.title(model_name)
                plt.scatter(x, predictions, color="yellow")
                plt.savefig(f"tmp/{model_name}.png")
                plt.close()

                # with open(model_path, "wb") as pkl:
                #     pickle.dump(model, pkl)
                # logging.info(f"Model saved as {model_name}")
