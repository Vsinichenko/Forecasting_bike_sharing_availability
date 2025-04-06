import os

sample_dir = "bash_scripts/multiple_tasks"
goal_dir = "bash_scripts/multiple_tasks/sarimax_calendar"
sample_path = os.path.join(sample_dir, "sample.sh")

with open(sample_path, "r") as f:
    script_content = f.read()


print(script_content)

part_ls = [1, 2]
city_ls = ["DD", "FB"]
dep_var_ls = ["demand", "supply"]

for city in city_ls:
    for part in part_ls:
        for dep_var in dep_var_ls:
            added_part = f" --city {city} --part {part} --dep-var {dep_var}"
            adj_script = script_content + added_part
            goal_file_path = os.path.join(goal_dir, f"sarimax_{city}_{part}_{dep_var}.sh")

            with open(goal_file_path, "w") as f:
                f.write(adj_script)
