import json
import logging
import os
import pickle
import warnings
from math import sqrt

import pandas as pd
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO)
warnings.simplefilter(action="ignore", category=FutureWarning)
logging.info("Reading data")

EXPERIMENT_NAME = "sarima"

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
test_DD_1 = df_DD_1.loc[flt].sort_values("datetime_hour")

flt = df_DD_2.datetime_hour.dt.date.isin(test_range_2_DD)
train_validation_DD_2 = df_DD_2.loc[~flt]
test_DD_2 = df_DD_2.loc[flt].sort_values("datetime_hour")

# FB
df_FB_1 = df_FB.loc[df_FB.datetime_hour.dt.date <= test_range_1_FB[-1]]
df_FB_2 = df_FB.loc[df_FB.datetime_hour.dt.date > test_range_1_FB[-1]]

flt = df_FB_1.datetime_hour.dt.date.isin(test_range_1_FB)
train_validation_FB_1 = df_FB_1.loc[~flt]
test_FB_1 = df_FB_1.loc[flt].sort_values("datetime_hour")

flt = df_FB_2.datetime_hour.dt.date.isin(test_range_2_FB)
train_validation_FB_2 = df_FB_2.loc[~flt]
test_FB_2 = df_FB_2.loc[flt].sort_values("datetime_hour")


df_helper = {"DD": df_DD, "FB": df_FB}
dep_var_helper = {"demand": "rent_count", "supply": "return_count"}
test_df_helper = {"DD": {1: test_DD_1, 2: test_DD_2}, "FB": {1: test_FB_1, 2: test_FB_2}}
train_validation_df_helper = {"DD": {1: train_validation_DD_1, 2: train_validation_DD_2}, "FB": {1: train_validation_FB_1, 2: train_validation_FB_2}}

logging.info("Computing RMSE")

rmse_collector = {}

for city in ["DD", "FB"]:
    for current_cell in df_helper[city].hex_id.unique():
        for part in [1, 2]:
            for dep_var in ["demand", "supply"]:
                model_name = f"sarima_{city}_{dep_var}_part_{part}_cell_{current_cell}.pkl"
                logging.info(f"Processing {model_name}")
                model_path = f"models/{model_name}"
                if not os.path.exists(model_path):
                    logging.error(f"Model {model_name} missing")
                    continue

                with open(model_path, "rb") as f:
                    model = pickle.load(f)

                dep_colname = dep_var_helper[dep_var]
                test_df = test_df_helper[city][part]
                train_validation_df = train_validation_df_helper[city][part]
                train_validation = train_validation_df[train_validation_df.hex_id == current_cell].set_index("datetime_hour")[dep_colname]

                test = test_df[test_df.hex_id == current_cell].set_index("datetime_hour")[dep_colname]
                test = test.asfreq("h")

                predictions = model.predict(n_periods=len(test))
                flt = predictions < 0
                if flt.any():
                    logging.warning(f"Negative predictions for {model_name}")
                    # set negative predictions to 0
                    predictions[flt] = train_validation.mean()

                rmse = sqrt(mean_squared_error(test, predictions))
                rmse_collector[model_name] = rmse

for key, value in rmse_collector.items():
    logging.info(f"{key}: {value}")


with open(f"rmse/{EXPERIMENT_NAME}.json", "w") as f:
    json.dump(rmse_collector, f, indent=4)
