import os
import pickle

mycell = "871f1b559ffffff"
dep_var = "demand"
city = "DD"
part = 1

EXPERIMENT_NAME = "sarimax_all"

model_name = f"{EXPERIMENT_NAME}_{city}_{dep_var}_part_{part}_cell_{mycell}.pkl"
model_dir = f"models/{EXPERIMENT_NAME}"

model_path = os.path.join(model_dir, model_name)
with open(model_path, "rb") as f:
    model_fit = pickle.load(f)

print(model_fit.summary())

latex_summary = model_fit.summary().as_latex()

with open(f"tmp/sample_model_summary_{EXPERIMENT_NAME}.tex", "w") as f:
    f.write(latex_summary)
