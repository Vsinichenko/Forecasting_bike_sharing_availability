from sklearn.metrics import mean_squared_error
from math import sqrt
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
from glob import glob
import seaborn as sns


EXPERIMENT_NAME = "sarimax_all_no_weekdays"
PLOT = True


model_names = glob(f"models/{EXPERIMENT_NAME}/*")

params = []
for model_name in model_names:
    with open(model_name, "rb") as f:
        model_fit = pickle.load(f)
    for coef_name in model_fit.params.index:
        if coef_name == "const":
            continue
        params.append({"Model": model_name, "Coefficient": coef_name, "p_value": model_fit.pvalues[coef_name], "Value": model_fit.params[coef_name]})


df_params = pd.DataFrame(params)


df_params["City"] = ""
df_params.loc[df_params.Model.str.contains("FB"), "City"] = "Freiburg"
df_params.loc[df_params.Model.str.contains("DD"), "City"] = "Dresden"


df_params = df_params.sort_values(by=["City", "Coefficient", "p_value"])

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
