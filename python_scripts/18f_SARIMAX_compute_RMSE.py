import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime
from math import sqrt

import pandas as pd
from sklearn.metrics import mean_squared_error

EXPERIMENT_NAME = "sarimax_all_optimized_adj_events_2"


warnings.simplefilter(action="ignore", category=FutureWarning)

parser = argparse.ArgumentParser()
# add default value
parser.add_argument("--depvar", type=str, choices=["demand", "supply", "demand_supply"], default="demand_supply", help="Dependent variable to predict")
parser.add_argument("--part", type=str, choices=["1_2", "1", "2"], default="1_2", help="Part")
parser.add_argument("--city", type=str, choices=["DD_FB", "DD", "FB"], default="DD_FB", help="City")
args = parser.parse_args()
dep_var_ls = ["demand", "supply"] if args.depvar == "demand_supply" else [args.depvar]
part_ls = [1, 2] if args.part == "1_2" else [int(args.part)]
city_ls = ["FB", "DD"] if args.city == "DD_FB" else [args.city]

mycell = "871f1b559ffffff"

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f"logs/{EXPERIMENT_NAME}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
model_dir = f"models/{EXPERIMENT_NAME}"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
img_dir = f"tmp/images/{EXPERIMENT_NAME}"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

log_fullpath = os.path.join(log_dir, f"model_{start_time}.log")

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

# bike trips
filename_DD = f"data/df_DD_for_SARIMAX_with_adj_events_2_2025-04-28_15-47-45.csv"
filename_FB = f"data/df_FB_for_SARIMAX_with_adj_events_2_2025-04-28_15-47-45.csv"
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

exog_colnames_base = [
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
    "is_dayoff",
    "Humidity",
]


if EXPERIMENT_NAME == "sarimax_all_no_weekdays":
    exog_colnames_base.append("Temperature", "Temperature_times_Humidity")


exog_colnames_demand = exog_colnames_base + ["event_count_end"]
exog_colnames_supply = exog_colnames_base + ["event_count_start"]

rmse_collector = {}

for city in city_ls:
    for current_cell in df_helper[city].hex_id.unique():
        for part in part_ls:
            for dep_var in dep_var_ls:
                if dep_var == "demand":
                    exog_colnames = exog_colnames_demand
                else:
                    exog_colnames = exog_colnames_supply

                model_name = f"{EXPERIMENT_NAME}_{city}_{dep_var}_part_{part}_cell_{current_cell}.pkl"
                logging.info(f"CITY {city} CURRENT CELL {current_cell}, PART {part}, DEPVAR {dep_var}")

                model_path = os.path.join(model_dir, model_name)
                if not os.path.exists(model_path):
                    logging.error(f"Model {model_name} does not exist")
                    continue

                with open(model_path, "rb") as f:
                    model_fit = pickle.load(f)

                dep_colname = dep_var_helper[dep_var]
                train_df = train_df_helper[city][part]
                train_df = train_df.loc[train_df.hex_id == current_cell].set_index("datetime_hour")

                test_df = test_df_helper[city][part]
                test_df = test_df.loc[test_df.hex_id == current_cell].set_index("datetime_hour")

                train_sr = train_df[dep_colname]

                test_exog_df = train_df[exog_colnames]

                test_sr = test_df[dep_colname]
                test_exog_df = test_df[exog_colnames]

                # print model summary
                predictions = model_fit.get_forecast(steps=len(test_sr), exog=test_exog_df).predicted_mean
                predictions[predictions < 0] = 0

                rmse = sqrt(mean_squared_error(test_sr, predictions))
                rmse_collector[model_name] = rmse


for key, value in rmse_collector.items():
    logging.info(f"{key}: {value}")

with open(f"rmse/{EXPERIMENT_NAME}.json", "w") as f:
    json.dump(rmse_collector, f, indent=4)
