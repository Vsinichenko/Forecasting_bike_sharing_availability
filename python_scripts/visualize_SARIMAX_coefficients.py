import argparse
import gc
import json
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from glob import glob
from math import sqrt

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

EXPERIMENT_NAME = "sarimax_all_optimized_adj_events"
PRINT_COEFS = True
PLOT = True


# obtain small hex ids
filename_DD = f"data/df_DD_for_SARIMAX_2025-04-08_14-28-37.csv"
filename_FB = f"data/df_FB_for_SARIMAX_2025-04-08_14-28-37.csv"
df_DD = pd.read_csv(filename_DD, index_col=None, parse_dates=["datetime_hour"])
df_FB = pd.read_csv(filename_FB, index_col=None, parse_dates=["datetime_hour"])
df_bike = pd.concat([df_DD, df_FB], axis=0, ignore_index=True)
hex_id_grouping = df_bike.groupby("hex_id")["rent_count"].sum()
small_hex_id_grouping = hex_id_grouping[hex_id_grouping < 5000]
small_hex_ids = small_hex_id_grouping.index.tolist()


model_names = glob(f"models/{EXPERIMENT_NAME}/*")

params = []
for model_name in model_names:
    for hex_id in small_hex_ids:
        if hex_id in model_name:
            continue

    with open(model_name, "rb") as f:
        model_fit = pickle.load(f)
    for coef_name in model_fit.params.index:
        if coef_name == "const":
            continue
        params.append({"Model": model_name, "Coefficient": coef_name, "p_value": model_fit.pvalues[coef_name], "Value": model_fit.params[coef_name]})


df_params = pd.DataFrame(params)

for h in range(1, 10):  # for correct sortings
    df_params["Coefficient"] = df_params["Coefficient"].str.replace(rf"\bhour_{h}\b", f"hour_0{h}", regex=True)


df_params["City"] = ""
df_params.loc[df_params.Model.str.contains("FB"), "City"] = "Freiburg"
df_params.loc[df_params.Model.str.contains("DD"), "City"] = "Dresden"


df_params = df_params.sort_values(by=["City", "Coefficient", "p_value"])

if PRINT_COEFS:
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", None, "display.float_format", "{:.10f}".format):
        print(df_params)


df_params.Coefficient.unique()


flt = df_params.Coefficient.isin(["ar.L1", "ma.L1", "ar.S.L24", "ma.S.L24", "sigma2"])

df_params = df_params[~flt]


df_params


if PLOT:
    for city in df_params.City.unique():
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_params.query(f"City=='{city}'"), x="Value", y="Coefficient")
        plt.axvline(0, color="grey", linestyle="--")
        plt.tight_layout()
        plt.savefig(f"tmp/images/{EXPERIMENT_NAME}_coefficients_{city}.png")
        plt.close()
