from sklearn.metrics import mean_squared_error
from math import sqrt

# import pmdarima.arima as pm_arima
import gc
import pandas as pd
import time
import pickle
import logging
import sys
from datetime import datetime
from matplotlib import pyplot as plt
import warnings
import argparse
import os
import json

EXPERIMENT_NAME = "sarimax_calendar_weather"

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

filename_DD = f"data/nextbike/hourly_demand_supply_Dresden 2025-03-19_10-47-56.csv"
filename_FB = f"data/nextbike/hourly_demand_supply_Freiburg_missing_interpolated_2025-03-19_10-47-56.csv"
df_DD = pd.read_csv(filename_DD, index_col=None, parse_dates=["datetime_hour"])
df_FB = pd.read_csv(filename_FB, index_col=None, parse_dates=["datetime_hour"])
df_DD = df_DD.sort_values("datetime_hour")
df_FB = df_FB.sort_values("datetime_hour")

filename_weather_DD = "data/weather/df_Dresden_weather_hourly 2025-03-28_20-51-37.csv"
filename_weather_FB = "data/weather/df_Freiburg_weather_hourly 2025-03-28_20-51-37.csv"

df_weather_DD = pd.read_csv(filename_weather_DD, index_col=None, parse_dates=["datetime_hour"])
df_weather_FB = pd.read_csv(filename_weather_FB, index_col=None, parse_dates=["datetime_hour"])

df_weather_DD = df_weather_DD.drop(columns=["Precipitation", "Wind"])
df_weather_FB = df_weather_FB.drop(columns=["Precipitation", "Wind"])

df_weather_DD["Temperature_times_Humidity"] = df_weather_DD["Temperature"] * df_weather_DD["Humidity"]
df_weather_FB["Temperature_times_Humidity"] = df_weather_FB["Temperature"] * df_weather_FB["Humidity"]


df_DD = df_DD.merge(df_weather_DD, on="datetime_hour", how="left")
df_FB = df_FB.merge(df_weather_FB, on="datetime_hour", how="left")


# test date ranges
test_range_1_DD = pd.date_range(start="2024-03-21", end="2024-03-31")
test_range_1_DD = [date.date() for date in test_range_1_DD]

test_range_2_DD = pd.date_range(start="2024-10-21", end="2024-10-31")
test_range_2_DD = [date.date() for date in test_range_2_DD]

test_range_1_FB = pd.date_range(start="2023-07-24", end="2023-07-31")
test_range_1_FB = [date.date() for date in test_range_1_FB]

test_range_2_FB = pd.date_range(start="2024-10-23", end="2024-10-31")
test_range_2_FB = [date.date() for date in test_range_2_FB]

## calendar effects
for i, df_tmp in enumerate([df_DD, df_FB]):
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
    if i == 0:
        # holidays for Dresden
        german_holidays = ["2024-01-01", "2024-03-29", "2024-04-01", "2024-05-01", "2024-05-09", "2024-05-20", "2024-10-03", "2024-10-31"]
    else:
        german_holidays = ["2023-06-08", "2024-10-03"]
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
    "Temperature",
    "Humidity",
    "Temperature_times_Humidity",
]

rmse_collector = {}

for city in city_ls:
    for current_cell in df_helper[city].hex_id.unique():
        for part in part_ls:
            for dep_var in dep_var_ls:
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
