from pmdarima import auto_arima
import pandas as pd
import time
import pickle
import logging
import sys
from datetime import datetime

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


# Load data
files_datetime = "2025-03-17_13-46-14"
filename_DD = f"data/nextbike/demand_supply_Dresden {files_datetime}.csv"
filename_FB = f"data/nextbike/demand_supply_Freiburg {files_datetime}.csv"
df_DD = pd.read_csv(filename_DD, index_col=None, parse_dates=["datetime_30min"])
df_FB = pd.read_csv(filename_FB, index_col=None, parse_dates=["datetime_30min"])

# Train ARIMA model for one hexagon
mycell = "871f1b559ffffff"
test_range_1_DD = pd.date_range(start="2024-03-21", end="2024-03-31")
test_range_2_DD = pd.date_range(start="2024-10-21", end="2024-10-31")

test_range_DD = test_range_1_DD.append(test_range_2_DD)

test_range_DD = [date.date() for date in test_range_DD]
train_validation_DD = df_DD.loc[~df_DD.datetime_30min.dt.date.isin(test_range_DD)].sort_values("datetime_30min")
test_DD = df_DD.loc[df_DD.datetime_30min.dt.date.isin(test_range_DD)].sort_values("datetime_30min")
train_DD = train_validation_DD[train_validation_DD.hex_id == mycell].set_index("datetime_30min")["rent_count"]

start_train_time = time.time()

model = auto_arima(y=train_DD, trace=True, suppress_warnings=False, seasonal=True, m=48)

logging.info(f"Elapsed time: {(time.time() - start_train_time)/60} minutes")

logging.info(f"Best ARIMA model: {model.summary()}")


with open(f"models/arima {start_time}.pkl", "wb") as pkl:
    pickle.dump(model, pkl)
