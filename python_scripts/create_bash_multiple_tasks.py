import os

EXPERIMENT_NAME = "sarimax_calendar_weather"

sample_dir = "bash_scripts/multiple_tasks"
sample_path = os.path.join(sample_dir, "sample.sh")

goal_dir = os.path.join(sample_dir, EXPERIMENT_NAME)
if not os.path.exists(goal_dir):
    os.makedirs(goal_dir)

with open(sample_path, "r") as f:
    script_content = f.read()


print(script_content)

part_ls = [1, 2]
city_ls = ["DD", "FB"]
depvar_ls = ["demand", "supply"]

for city in city_ls:
    for part in part_ls:
        for depvar in depvar_ls:
            added_part = f" --city {city} --part {part} --depvar {depvar}"
            adj_script = script_content + added_part
            current_task_name = f"{EXPERIMENT_NAME}_{city}_{part}_{depvar}"
            adj_script = adj_script.replace("TASK_NAME", current_task_name)

            goal_file_path = os.path.join(goal_dir, f"{EXPERIMENT_NAME}_{current_task_name}.sh")

            with open(goal_file_path, "w") as f:
                f.write(adj_script)
