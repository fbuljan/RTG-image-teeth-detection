import os
import re
import numpy as np

main_folder_name = "detectron2_output"

metrics = {}
main_folder_path = os.path.join(".", main_folder_name)

for subdir, _, files in os.walk(main_folder_path):
    if "results.txt" in files:
        path = os.path.join(subdir, "results.txt")
        with open(path, "r") as f:
            for line in f:
                match = re.match(r"([\w_.]+):\s+([0-9.]+)", line)
                if match:
                    key, value = match.group(1), float(match.group(2))
                    metrics.setdefault(key, []).append(value)

output_path = os.path.join(".", "results_combined.txt")
with open(output_path, "w") as out:
    for key in sorted(metrics):
        values = np.array(metrics[key])
        avg = np.mean(values)
        std = np.std(values)
        out.write(f"{key}: avg = {avg:.4f}, std = {std:.4f}\n")

print(f"Izračunano {len(metrics)} metrika. Rezultati su zapisani u: {output_path}")
