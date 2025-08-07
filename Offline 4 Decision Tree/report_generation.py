import subprocess
import csv

# Compile both C++ files
compile_result = subprocess.run(["g++", "dataset_divider.cpp", "-o", "dataset_divider.exe"])
another_compile_result = subprocess.run(["g++", "task1.cpp", "-o", "iris_task.exe"])

# Store results here as a list of dicts
results = []

# Mapping criteria to enforce sort order
criterion_order = {"ig": 0, "igr": 1, "ngiw": 2}

# Proceed only if both compilations succeed
if compile_result.returncode == 0 and another_compile_result.returncode == 0:
    for i in range(20):
        print(f"\n Run {i+1}/20")

        # Run dataset divider
        subprocess.run(["dataset_divider.exe", "1"])

        # Set depth
        if i < 6:
            d = 3
        elif i < 12:
            d = 6
        else:
            d = 9

        # Run all 3 criteria
        for criterion_flag in ["ig", "igr", "ngiw"]:
            subprocess.run(["iris_task.exe", criterion_flag, str(d)])

            # Defaults
            criterion = ""
            accuracy = 0.0
            depth = 0
            real_depth = 0
            node_count = 0

            # Read prediction results
            with open("predictions.csv", "r") as file:
                for line in file:
                    if line.startswith("Criterion:"):
                        criterion = line.split(":")[1].strip()
                    elif line.startswith("Accuracy:"):
                        accuracy = float(line.split(":")[1].strip().rstrip('%'))
                    elif line.startswith("Depth:"):
                        parts = line.split(":")[1].strip().split(",")
                        depth = int(parts[0])
                        real_depth = int(parts[1])
                    elif line.startswith("Node:"):
                        node_count = int(line.split(":")[1].strip())

            # Store each run as a dict
            results.append({
                "criterion": criterion,
                "depth": d,  # use loop depth, not parsed value
                "accuracy": accuracy,
                "node_count": node_count,
                "real_depth": real_depth
            })

    #  Sort results by criterion and then depth
    results.sort(key=lambda x: (criterion_order.get(x["criterion"], 99), x["depth"]))

    #  Write to CSV
    with open("iris_report.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Criterion", "Depth", "Accuracy", "Node Count", "Real Depth"])
        for r in results:
            writer.writerow([r["criterion"], r["depth"], r["accuracy"], r["node_count"], r["real_depth"]])

    print("\n All runs complete. Sorted results saved to 'iris_report.csv'.")

else:
    print(" Compilation failed.")
