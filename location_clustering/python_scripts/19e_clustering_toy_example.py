import os
import pickle

# import contextily as ctx
import h3
import pandas as pd
from libpysal.weights import W
from spopt.region import RegionKMeansHeuristic

# def plot_df(df, column=None, ax=None, add_basemap=True):
#     "Plot based on the `geometry` column of a GeoPandas dataframe"
#     df = df.copy()
#     df = df.to_crs(epsg=3857)  # web mercator

#     if ax is None:
#         _, ax = plt.subplots(figsize=(8, 8))
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     df.plot(
#         ax=ax,
#         alpha=0.25,
#         edgecolor="k",
#         column=column,
#         categorical=True,
#         legend=True,
#         legend_kwds={"loc": "upper left"},
#     )
#     if add_basemap:
#         ctx.add_basemap(ax, crs=df.crs, source=ctx.providers.CartoDB.Positron)


# def plot_shape(shape, ax=None, add_basemap=True):
#     df = gpd.GeoDataFrame({"geometry": [shape]}, crs="EPSG:4326")
#     plot_df(df, ax=ax, add_basemap=add_basemap)


# def plot_cell(cell, ax=None):
#     shape = h3.cells_to_h3shape([cell])
#     plot_shape(shape, ax=ax)


# def plot_cells(cells, ax=None):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     shape = h3.cells_to_h3shape(cells)
#     plot_shape(shape, ax=ax, add_basemap=True)

#     for single_cell in cells:
#         single_shape = h3.cells_to_h3shape([single_cell])
#         # gdf = gpd.GeoDataFrame({'geometry': [single_shape]}, crs='EPSG:4326')
#         # gdf = gdf.to_crs(epsg=3857)
#         # gdf.plot(ax=ax, alpha=0.5, edgecolor='k')
#         plot_shape(single_shape, ax=ax, add_basemap=False)


# def plot_cell_area(cells, save=False, img_name="tmp"):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     shape = h3.cells_to_h3shape(cells)
#     plot_shape(shape, ax=ax, add_basemap=True)
#     if save:
#         plt.savefig(f"tmp/images/{img_name}.png", dpi=300, bbox_inches="tight")


filename_DD = "data/nextbike/trips_DD_with_small_hexids_res10_2025-04-21_11-55-31.csv"
filename_FB = "data/nextbike/trips_FB_with_small_hexids_res10_2025-04-21_11-55-31.csv"
df_DD = pd.read_csv(filename_DD, index_col=0)
df_FB = pd.read_csv(filename_FB, index_col=0)


def transform_df(df_input):
    df = df_input.copy()
    df["hour"] = pd.to_datetime(df["datetime_rent"]).dt.hour
    df["weekday"] = pd.to_datetime(df["datetime_rent"]).dt.weekday
    df = df[df.weekday <= 4]
    df["hour_interval"] = df["hour"]  # pd.cut(df["hour"], bins=[0, 7, 12, 15, 20, 24], labels=["0-6", "7-11", "12-14", "15-19", "20-23"], right=False)
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


# df_DD_grouped = transform_df(df_DD)
# df_DD_grouped_scaled = scale_df(df_DD_grouped)

df_FB_grouped = transform_df(df_FB)
df_FB_grouped_scaled = scale_df(df_FB_grouped)

# my_neighbour = "8a1f8024429ffff"
# assert my_neighbour not in df_FB_grouped_scaled.index


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

    for cell in existing_hex_ids:
        neighbors = h3.grid_ring(cell, 1)
        for neighbor in neighbors:
            if neighbor not in existing_hex_ids:
                row_to_add = pd.DataFrame(columns=df_tmp.columns, index=[neighbor])
                # fill the row with values from the cell
                row_to_add.iloc[0, :] = df_tmp.loc[cell, :]
                df_tmp = pd.concat([df_tmp, row_to_add], axis=0)

    print(len(df_tmp))
    return df_tmp


# df_DD_grouped_scaled = add_missing_hex_ids(df_DD_grouped_scaled, df_DD)


df_FB_grouped_scaled = add_missing_hex_ids(df_FB_grouped_scaled, df_FB)

# print(df_FB_grouped_scaled.loc[my_neighbour])


# plot_cell_area(df_FB_grouped_scaled.index.tolist(), save=True, img_name="FB_cells")
# plot_cell_area(df_DD_grouped_scaled.index.tolist(), save=True, img_name = "DD_cells")


existing_hex_ids = set(df_FB_grouped_scaled.index)


START_NEIGHBOURS_DISTANCE = 5  # from the visual analysis of maps, because there are "islands" of several cells


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


len(neighbors_dict)

df_FB_grouped_scaled.sort_index(inplace=True)

toy_data = df_FB_grouped_scaled.iloc[:100]
toy_hex_ids = toy_data.index.tolist()

toy_neighbors_dict = {hex_id: neighbors_dict[hex_id] for hex_id in toy_hex_ids}
toy_neighbors_dict = {hex_id: [cell for cell in toy_neighbors_dict[hex_id] if cell in toy_hex_ids] for hex_id in toy_hex_ids}

toy_w = W(toy_neighbors_dict)


toy_w = toy_w.symmetrize()


import time

model = RegionKMeansHeuristic(data=toy_data, n_clusters=25, w=toy_w, drop_islands=False)

start_time = time.time()
model.solve()
end_time = time.time()
print(f"Time taken to solve the model: {end_time - start_time} seconds")

model_dir = "models/clustering/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, "less_complex_model_FB.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)
