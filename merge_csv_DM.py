import pandas as pd

# Load both CSVs
dm_df = pd.read_csv("../DM_breast.csv")
train_df = pd.read_csv("../train_undersampling_breast.csv")

# Filter for positive samples with both views
dm_filtered = dm_df[
    (dm_df["cancer"] == 1) &
    (dm_df["image_id_CC"].notna()) & (dm_df["image_id_CC"].str.strip() != "") &
    (dm_df["image_id_MLO"].notna()) & (dm_df["image_id_MLO"].str.strip() != "")
].copy()

print(len(dm_filtered), "positive samples with both views")

# Define all required columns in final output
required_columns = train_df.columns.tolist()

# Fill missing columns with default values
for col in required_columns:
    if col not in dm_filtered.columns:
        if col in ["laterality", "density", "machine_id"]:
            dm_filtered[col] = "X"  # placeholder string
        elif col in ["age", "biopsy", "invasive", "BIRADS", "implant", "difficult_negative_case"]:
            dm_filtered[col] = 0
        else:
            dm_filtered[col] = None  # fallback for unexpected columns

# Reorder columns to match train_df
dm_filtered = dm_filtered[required_columns]

# Concatenate and save
combined_df = pd.concat([train_df, dm_filtered], ignore_index=True)
combined_df.to_csv("../train_undersampling_breast_DM.csv", index=False)