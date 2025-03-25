import os
import pickle

model_names = os.listdir("models")
model_names = [name for name in model_names if name.endswith(".pkl")]

for model_name in model_names:
    model_path = os.path.join("models", model_name)
    with open(model_path, "rb") as f:
        print(model_path)

        model = pickle.load(f)
        order = model.order  # (p, d, q)
        seasonal_order = model.seasonal_order  # (P, D, Q, s)

        summary_str = f"ARIMA{order}{seasonal_order}"
        print(summary_str)
