# %%
import pickle

# import auto_arima # requires numpy version 1.26.4 i.e. before 2.
import time

import pandas as pd
import pmdarima as pm
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

# %%
datetime = "2025-03-19_10-47-56"
filename_DD = f"data/nextbike/hourly_demand_supply_Dresden {datetime}.csv"
filename_FB = f"data/nextbike/hourly_demand_supply_Freiburg {datetime}.csv"
df_DD = pd.read_csv(filename_DD, index_col=None, parse_dates=["datetime_hour"])
df_FB = pd.read_csv(filename_FB, index_col=None, parse_dates=["datetime_hour"])

# %%
mycell = "871f1b559ffffff"
mycell = "871f1b54bffffff"

# %%
df_DD.head(2)

# %%
# # df_DD.groupby("hex_id").agg({'rent_count': 'sum',
#                                 'return_count': 'sum'})


# %%
# df_DD.rent_count.sum()

# %%
# df_DD.hex_id.unique()

# %%
df_FB.hex_id.unique()

# %%
df_DD.loc[df_DD.hex_id == mycell, ["datetime_hour", "rent_count", "return_count"]].plot(x="datetime_hour", y=["rent_count", "return_count"])

# %%
test_range_1_DD = pd.date_range(start="2024-03-21", end="2024-03-31")
test_range_1_DD = [date.date() for date in test_range_1_DD]

# %%
test_range_2_DD = pd.date_range(start="2024-10-21", end="2024-10-31")
test_range_2_DD = [date.date() for date in test_range_2_DD]


# %%
df_DD_1 = df_DD.loc[df_DD.datetime_hour.dt.date <= test_range_1_DD[-1]]

# %%
df_DD_2 = df_DD.loc[df_DD.datetime_hour.dt.date > test_range_1_DD[-1]]

# %%
train_validation_DD_1 = df_DD_1.loc[~df_DD_1.datetime_hour.dt.date.isin(test_range_1_DD)]

# %%
test_DD_1 = df_DD.loc[df_DD.datetime_hour.dt.date.isin(test_range_1_DD)].sort_values("datetime_hour")

# %%
train_validation_DD_2 = df_DD_2.loc[~df_DD_2.datetime_hour.dt.date.isin(test_range_2_DD)].sort_values("datetime_hour")

# %%
test_DD_2 = df_DD_2.loc[df_DD_2.datetime_hour.dt.date.isin(test_range_2_DD)].sort_values("datetime_hour")

# %%
len(test_DD_2)

# %%
len(train_validation_DD_2)

# %%
train_1 = train_validation_DD_1[train_validation_DD_1.hex_id == mycell].set_index("datetime_hour")["rent_count"]

# %%
test_1 = test_DD_1[test_DD_1.hex_id == mycell].set_index("datetime_hour")["rent_count"]

# %%
train_1 = train_1.asfreq("h")

# %%
# test_1.asfreq('h').isna().value_counts()

# %%
test_1.asfreq("h").isna().value_counts()

# %%
test_1 = test_1.asfreq("h")

# %%
model_name = f"sarima_DD_deman_part_1_cell_{mycell}.pkl"
with open(f"models/model_name", "rb") as f:
    model = pickle.load(f)

# %%
print(model.summary())


# %%
predictions = model.predict(n_periods=len(test_1))
slize_size = 24 * 5

x = (test_1.index)[:slize_size]
plt.plot(x, test_1.iloc[:slize_size], color="blue")
plt.scatter(x, predictions.iloc[:slize_size], color="yellow")
plt.show()

# %%
mycell

# %%
plt.figure(figsize=(10, 5))
sns.lineplot(data=test_1, label="Test data")
sns.lineplot(data=predictions, label="Predictions", linestyle="--")
plt.xlabel("Datetime hour")
plt.ylabel("Rent count")
# plt.title(f"Rents and returns by minute on {day_str}")
plt.xticks(rotation=90)
plt.legend()
plt.show()

# %%
mean_squared_error(test_1, predictions)

# %%


# %% [markdown]
# # try to fit a model with a 0 ar component

# %%
# model = pm.auto_arima(train_1, start_p=0, max_p=0, d=1, seasonal=True, m=24, stepwise=True, suppress_warnings=True, error_action="ignore")

# %%

# %%
# with open(model_name, "wb") as pkl:
#     pickle.dump(model, pkl)

# %%
model.summary()

# %%
predictions = model.predict(n_periods=len(test_1))
slize_size = 24 * 5

x = (test_1.index)[:slize_size]
plt.plot(x, test_1.iloc[:slize_size], color="blue")
plt.scatter(x, predictions.iloc[:slize_size], color="yellow")
plt.show()
plt.savefig(f"tmp/{model_name}_preict_vs_fact.png", dpi=300, bbox_inches="tight")
