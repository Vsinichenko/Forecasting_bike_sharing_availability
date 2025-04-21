import os
import pickle

# import geopandas as gpd
import h3

# import matplotlib.pyplot as plt
import pandas as pd
from libpysal.weights import W
from spopt.region import RegionKMeansHeuristic

filename_DD = "data/nextbike/trips_DD_with_small_hexids_res8_2025-04-21_17-15-43.csv"
filename_FB = "data/nextbike/trips_FB_with_small_hexids_res8_2025-04-21_17-15-43.csv"
df_DD = pd.read_csv(filename_DD, index_col=0)
df_FB = pd.read_csv(filename_FB, index_col=0)

START_NEIGHBOURS_DISTANCE = 1


def transform_df(df_input):
    df = df_input.copy()
    df["hour"] = pd.to_datetime(df["datetime_rent"]).dt.hour
    df["weekday"] = pd.to_datetime(df["datetime_rent"]).dt.weekday
    df = df[df.weekday <= 4]
    df["hour_interval"] = pd.cut(df["hour"], bins=[0, 7, 12, 15, 20, 24], labels=["0-6", "7-11", "12-14", "15-19", "20-23"], right=False)  # df["hour"]
    df_grouped = df.groupby(["small_hex_id_rent", "hour_interval"]).size()
    df_grouped = df_grouped.reset_index(name="count_rent")
    df_grouped = df_grouped.pivot(index="small_hex_id_rent", columns="hour_interval", values="count_rent").fillna(0).astype(int)
    return df_grouped


def scale_df(df_grouped):
    df_grouped_tmp = df_grouped.copy()

    def min_max_scale(x):
        denom = x.max() - x.min()
        if denom == 0:
            return x
        else:
            return (x - x.min()) / denom

    df_grouped_scaled_tmp = df_grouped_tmp.apply(lambda x: min_max_scale(x), axis=1)
    df_grouped_scaled_tmp["total_count"] = df_grouped_tmp.apply(sum, axis=1)
    df_grouped_scaled_tmp.total_count = min_max_scale(df_grouped_scaled_tmp.total_count)
    return df_grouped_scaled_tmp


df_FB_grouped = transform_df(df_FB)
df_FB_grouped_scaled = scale_df(df_FB_grouped)


def add_missing_hex_ids(df_grouped_scaled_input, df_input):
    df_tmp = df_grouped_scaled_input.copy()
    print(len(df_tmp))

    existing_hex_ids = df_tmp.index.tolist()
    to_add = set(df_input.loc[~df_input.small_hex_id_rent.isin(existing_hex_ids)].small_hex_id_rent.dropna().unique().tolist())
    len(to_add)
    to_add = to_add | set(df_input.loc[~df_input.small_hex_id_return.isin(existing_hex_ids)].small_hex_id_return.dropna().unique().tolist())
    len(to_add)
    rows_to_add = pd.DataFrame(columns=df_tmp.columns, index=list(to_add))
    rows_to_add.fillna(0, inplace=True)
    df_tmp = pd.concat([df_tmp, rows_to_add], axis=0)

    print(len(df_tmp))
    return df_tmp


df_FB_grouped_scaled = add_missing_hex_ids(df_FB_grouped_scaled, df_FB)
existing_hex_ids = set(df_FB_grouped_scaled.index)
existing_hex_ids = set(df_FB_grouped_scaled.index)
neighbors_dict = {hex_id: [cell for cell in h3.grid_ring(hex_id, START_NEIGHBOURS_DISTANCE) if cell in existing_hex_ids] for hex_id in existing_hex_ids}


def count_islands(neighbors_dict):
    counter = 0
    for cell in neighbors_dict.keys():
        if len(neighbors_dict[cell]) == 0:
            counter += 1
    return counter


neighbours_distance = START_NEIGHBOURS_DISTANCE

while True:
    islands = count_islands(neighbors_dict)
    print(f"{islands=}")
    if islands > 0:
        neighbours_distance += 1
        print(f"{neighbours_distance=}")
        for cell in neighbors_dict.keys():
            if len(neighbors_dict[cell]) == 0:
                neighbors_dict[cell] = [cell for cell in h3.grid_ring(cell, neighbours_distance) if cell in existing_hex_ids]
    else:
        break


w = W(neighbors_dict)
w = w.symmetrize()
print(f"{w.n_components=}")

print(f"{len(df_FB_grouped_scaled)=}")

model = RegionKMeansHeuristic(data=df_FB_grouped_scaled, n_clusters=25, w=w, drop_islands=True)

print("Start clustering")
model.solve()
print("Finish clustering")

model_dir = "models/clustering/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, "less_complex_kmeans_model_FB.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)
