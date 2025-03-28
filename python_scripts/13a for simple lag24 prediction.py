import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import json
import logging
import warnings
import seaborn as sns
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)
warnings.simplefilter(action="ignore", category=FutureWarning)
logging.info("Reading data")

mycell = "871f1b559ffffff"

EXPERIMENT_NAME = "simple_HA_hourly"

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
                model_name = f"{city}_{dep_var}_part_{part}_cell_{current_cell}.pkl"
                dep_colname = dep_var_helper[dep_var]

                test_df = test_df_helper[city][part]
                train_validation_df = train_validation_df_helper[city][part]

                test = test_df[test_df.hex_id == current_cell].set_index("datetime_hour")[dep_colname]
                train_validation = train_validation_df[train_validation_df.hex_id == current_cell].set_index("datetime_hour")[dep_colname]

                if city == "FB" and part == 1:
                    train_validation = train_validation.asfreq("h", fill_value=train_validation.mean())
                else:
                    train_validation = train_validation.asfreq("h")

                test = test.asfreq("h")

                n_steps = len(test)
                predictions = pd.Series(index=test.index, dtype=float)
                for hour in range(24):
                    avg_houly_value = train_validation.loc[train_validation.index.hour == hour].mean()
                    predictions.loc[predictions.index.hour == hour] = avg_houly_value

                rmse = sqrt(mean_squared_error(test, predictions))
                rmse_collector[model_name] = rmse

                if city == "DD" and part == 1 and current_cell == mycell:
                    plt.figure(figsize=(10, 5))
                    sns.lineplot(data=test, label="Test data")
                    sns.lineplot(data=predictions, label="Predictions", linestyle="--")
                    plt.xlabel("Datetime hour")
                    plt.ylabel("Rent count")
                    # plt.title(f"Rents and returns by minute on {day_str}")
                    plt.xticks(rotation=90)
                    plt.savefig(f"tmp/sample_lag24_prediction.png", bbox_inches="tight")
                    plt.legend()
                    plt.show()


for key, value in rmse_collector.items():
    logging.info(f"{key}: {value}")


with open(f"rmse/{EXPERIMENT_NAME}.json", "w") as f:
    json.dump(rmse_collector, f, indent=4)
