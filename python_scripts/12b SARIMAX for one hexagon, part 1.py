#!/usr/bin/env python
# coding: utf-8


from pmdarima import auto_arima
import pmdarima as pm
import pandas as pd
import time
import pickle
import logging
import sys
from datetime import datetime
from matplotlib import pyplot as plt

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_fullpath = f"../output_{start_time}.log"

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
filename_DD = f"../data/nextbike/hourly_demand_supply_Dresden {file_datetime}.csv"
filename_FB = f"../data/nextbike/hourly_demand_supply_Freiburg {file_datetime}.csv"
df_DD = pd.read_csv(filename_DD, index_col=None, parse_dates=["datetime_hour"])
df_FB = pd.read_csv(filename_FB, index_col=None, parse_dates=["datetime_hour"])


mycell = "871f1b559ffffff"


df_DD.rent_count.sum()


df_DD.hex_id.unique()


df_FB.hex_id.unique()


df_DD.loc[df_DD.hex_id == mycell, ["datetime_hour", "rent_count", "return_count"]].plot(
    x="datetime_hour", y=["rent_count", "return_count"]
)


test_range_1_DD = pd.date_range(start="2024-03-21", end="2024-03-31")
test_range_1_DD = [date.date() for date in test_range_1_DD]


test_range_2_DD = pd.date_range(start="2024-10-21", end="2024-10-31")
test_range_2_DD = [date.date() for date in test_range_2_DD]


df_DD_1 = df_DD.loc[df_DD.datetime_hour.dt.date <= test_range_1_DD[-1]]


# df_DD_1.plot(x='datetime_hour', y=['rent_count', 'return_count'])


df_DD_2 = df_DD.loc[df_DD.datetime_hour.dt.date > test_range_1_DD[-1]]


train_validation_DD_1 = df_DD_1.loc[
    ~df_DD_1.datetime_hour.dt.date.isin(test_range_1_DD)
]


test_DD_1 = df_DD.loc[df_DD.datetime_hour.dt.date.isin(test_range_1_DD)].sort_values(
    "datetime_hour"
)


train_validation_DD_2 = df_DD_2.loc[
    ~df_DD_2.datetime_hour.dt.date.isin(test_range_2_DD)
].sort_values("datetime_hour")


test_DD_2 = df_DD_2.loc[
    df_DD_2.datetime_hour.dt.date.isin(test_range_2_DD)
].sort_values("datetime_hour")


len(test_DD_2)


len(train_validation_DD_2)


train_1 = train_validation_DD_1[train_validation_DD_1.hex_id == mycell].set_index(
    "datetime_hour"
)["rent_count"]


test_1 = test_DD_1[test_DD_1.hex_id == mycell].set_index("datetime_hour")["rent_count"]


train_1 = train_1.asfreq("h")


# test_1.asfreq('h').isna().value_counts()


test_1.asfreq("h").isna().value_counts()


test_1 = test_1.asfreq("h")


start_train_time = time.time()
logging.info("Start ARIMA optimisation")

model_1 = auto_arima(y=train_1, trace=True, suppress_warnings=False, seasonal=True, m=24, n_jobs=-1)
model_1.fit(train_1)

logging.info(f"Elapsed time: {(time.time() - start_train_time)/60} minutes")


logging.info(model_1.summary())


predictions = model_1.predict(n_periods=len(test_1))


x = test_1.index
plt.plot(x, test_1, color="black")
plt.scatter(x, predictions, color="black")
plt.show()


with open(f"models/arima DD part 1 mycell.pkl", "wb") as pkl:
    pickle.dump(model_1, pkl)