import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

EXPERIMENT_NAME = "sarimax_all_no_weekdays_only_humidity"

PLOTS = True


with open(f"rmse/{EXPERIMENT_NAME}.json", "r") as f:
    rmse = json.load(f)

for key in rmse.keys():
    if rmse[key] > 30:
        print(key)
        print(rmse[key])


for phrase in ["DD_demand_", "FB_demand_", "DD_supply_", "FB_supply_"]:
    new_index = [key for key in rmse.keys() if phrase in key]
    # sort list alphabetically
    new_index.sort()
    new_values = [rmse[key] for key in new_index]
    new_index = [key.replace(phrase, "") for key in new_index]

    myseries = pd.Series(index=new_index, data=new_values)

    if PLOTS:
        plt.figure(figsize=(10, 5))
        sns.barplot(myseries)
        plt.xlabel("DD_demand")
        plt.ylabel("RMSE")
        plt.title(f"RMSE for {EXPERIMENT_NAME} for {phrase}")

        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    print(phrase)
    print(int(myseries.sum()))
