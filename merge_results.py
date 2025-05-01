import pandas as pd

# Load each CSV
sgd = pd.read_csv("results_sgd.csv")
adam = pd.read_csv("results_adam.csv")
rmsprop = pd.read_csv("results_rmsprop.csv")

# Add optimizer column if not present
if "Optimizer" not in sgd.columns:
    sgd["Optimizer"] = "SGD"
if "Optimizer" not in adam.columns:
    adam["Optimizer"] = "Adam"
if "Optimizer" not in rmsprop.columns:
    rmsprop["Optimizer"] = "RMSprop"

# Reorder to keep consistent format
columns_order = ['Run', 'Optimizer', 'LR', 'Batch Size', 'Dropout', 'Batch Norm', 'Final Accuracy']
sgd = sgd[columns_order]
adam = adam[columns_order]
rmsprop = rmsprop[columns_order]

# Adjust run numbers to stay unique
adam["Run"] += sgd["Run"].max()
rmsprop["Run"] += adam["Run"].max()

# Merge all three
merged = pd.concat([sgd, adam, rmsprop], ignore_index=True)

# Save the merged CSV
merged.to_csv("merged_results.csv", index=False)
print("Merged CSV saved as 'merged_results.csv'")
