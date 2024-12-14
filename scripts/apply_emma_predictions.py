import pandas as pd

# File paths
file_with_invalid = "emma_annotations/emma_training_set_annotations.csv"
file_without_invalid = "annotations/completed_data_fixed_10_epochs.csv"

# Read datasets
df_invalid = pd.read_csv(file_with_invalid)
df_valid = pd.read_csv(file_without_invalid)

# Replace invalid values (10) with corresponding values from the valid dataset
df_cleaned = df_invalid.where(df_invalid != 10, df_valid)

# Save the cleaned dataset
df_cleaned.to_csv("annotations/final_emma_dataset_10_epochs.csv", index=False)
