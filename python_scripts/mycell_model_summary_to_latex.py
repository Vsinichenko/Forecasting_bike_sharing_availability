import os
import pickle

mycells = ["871f1b559ffffff"]
part = 1

TO_LATEX = True

EXPERIMENT_NAME = "sarimax_all_optimized_adj_events"

for mycell in mycells:
    for dep_var in ["demand", "supply"]:
        for city in ["DD"]:
            try:
                model_name = f"{EXPERIMENT_NAME}_{city}_{dep_var}_part_{part}_cell_{mycell}.pkl"
                print(model_name)
                model_dir = f"models/{EXPERIMENT_NAME}"

                model_path = os.path.join(model_dir, model_name)
                with open(model_path, "rb") as f:
                    model_fit = pickle.load(f)

                print(model_fit.summary())
            except FileNotFoundError:
                print(f"Model file not found: {model_path}")
                continue

            if TO_LATEX:
                latex_summary = model_fit.summary().as_latex()

                with open(f"tmp/{model_name}.tex", "w") as f:
                    f.write(latex_summary)
