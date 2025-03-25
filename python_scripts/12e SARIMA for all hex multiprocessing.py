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
import warnings
import os
import gc

warnings.simplefilter(action="ignore", category=FutureWarning)

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_fullpath = f"logs/all_hexagons_arima_multi_{start_time}.log"

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


def process_iteration(city, current_cell, part, dep_var, df):
    logging.info("Reading data")
    if city == "DD":
        test_range_1 = pd.date_range(start="2024-03-21", end="2024-03-31")
        test_range_2 = pd.date_range(start="2024-10-21", end="2024-10-31")
    elif city == "FB":
        test_range_1 = pd.date_range(start="2023-07-24", end="2023-07-31")
        test_range_2 = pd.date_range(start="2024-10-23", end="2024-10-31")

    test_range_1 = [date.date() for date in test_range_1]
    test_range_2 = [date.date() for date in test_range_2]
    df_1 = df.loc[df.datetime_hour.dt.date <= test_range_1[-1]]
    df_2 = df.loc[df.datetime_hour.dt.date > test_range_1[-1]]
    flt = df_1.datetime_hour.dt.date.isin(test_range_1)
    train_validation_1 = df_1.loc[~flt]
    test_1 = df_1.loc[flt].sort_values("datetime_hour")

    flt = df_2.datetime_hour.dt.date.isin(test_range_2)
    train_validation_2 = df_2.loc[~flt].sort_values("datetime_hour")
    test_2 = df_2.loc[flt].sort_values("datetime_hour")

    dep_var_helper = {"demand": "rent_count", "supply": "return_count"}
    train_df_helper = {1: train_validation_1, 2: train_validation_2}
    test_df_helper = {1: test_1, 2: test_2}

    model_name = f"sarima_{city}_{dep_var}_part_{part}_cell_{current_cell}.pkl"
    model_path = f"models/{model_name}"
    if os.path.exists(model_path):
        logging.info(f"Model {model_name} already exists. Skipping.")
        return

    logging.info(f"CITY {city} CURRENT CELL {current_cell}, PART {part}, DEPVAR {dep_var}")
    dep_colname = dep_var_helper[dep_var]
    train_df = train_df_helper[part]
    test_df = test_df_helper[part]
    train = train_df[train_df.hex_id == current_cell].set_index("datetime_hour")[dep_colname]
    test = test_df[test_df.hex_id == current_cell].set_index("datetime_hour")[dep_colname]

    if city == "FB" and part == 1:
        train = train.asfreq("h", fill_value=train.mean())
    else:
        train = train.asfreq("h")

    test = test.asfreq("h")

    start_train_time = time.time()
    logging.info("Start ARIMA optimisation")

    model = auto_arima(y=train, trace=True, stepwise=True, suppress_warnings=False, seasonal=True, m=24, d=1, D=0, max_p=3, max_q=2)

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

    with open(model_path, "wb") as pkl:
        pickle.dump(model, pkl)
    logging.info(f"Model saved as {model_name}")
    del model, trian, test, train_df, test_df
    gc.collect()


if __name__ == "__main__":
    import multiprocessing as mp

    num_cpus = mp.cpu_count()
    logging.info(f"Number of CPUs: {num_cpus}")

    logging.info("loading data")
    file_datetime = "2025-03-19_10-47-56"
    filename_FB = f"data/nextbike/hourly_demand_supply_Freiburg {file_datetime}.csv"
    df_FB = pd.read_csv(filename_FB, index_col=None, parse_dates=["datetime_hour"])

    filename_DD = f"data/nextbike/hourly_demand_supply_Dresden {file_datetime}.csv"
    df_DD = pd.read_csv(filename_DD, index_col=None, parse_dates=["datetime_hour"])
    df_helper = {"DD": df_DD, "FB": df_FB}

    tasks = [
        (city, current_cell, part, dep_var, df)
        for city in ["DD", "FB"]
        for current_cell in df_helper[city].hex_id.unique()
        for part in [1, 2]
        for dep_var in ["demand", "supply"]
        for df in df_helper[city]
    ]

    tasks_DD = [("DD", current_cell, part, dep_var, df_DD) for current_cell in df_DD.hex_id.unique() for part in [1, 2] for dep_var in ["demand", "supply"]]

    logging.info(f"{len(tasks_DD)} tasks for DD")
    tasks_FB = [("FB", current_cell, part, dep_var, df_FB) for current_cell in df_FB.hex_id.unique() for part in [1, 2] for dep_var in ["demand", "supply"]]

    logging.info(f"{len(tasks_FB)} tasks for FB")
    tasks = tasks_DD + tasks_FB
    logging.info(f"{len(tasks)} tasks in total")

    assert len(tasks) == 220, f"Incorrect number of tasks {len(tasks)}"

    with mp.Pool(2) as pool:
        pool.starmap(process_iteration, tasks)
