import pandas as pd
import numpy as np

# Load the original CSV
df = pd.read_csv('../../train.csv')

# Step 1: Replace string booleans with integers
df = df.replace({'False': 0, 'True': 1})

# Step 2: Convert actual bool dtype columns to integers
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

# Step 3: Force convert specific columns to int (if not already)
for col in ['implant', 'difficult_negative_case']:
    if col in df.columns:
        df[col] = df[col].astype(int)

# Step 4: Fill missing numeric values with column mean (except BIRADS)
for col in ['age']:
    if col in df.columns:
        num_missing = df[col].isna().sum()
        if num_missing > 0:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"Filled {num_missing} missing values in column '{col}' with mean: {mean_val:.2f}")

# Step 5: One-hot encode categorical columns
categorical_cols = ['site_id', 'laterality', 'view', 'machine_id']
df = pd.get_dummies(df, columns=categorical_cols, prefix=[f"{col}_id" if col == "site_id" else col for col in categorical_cols])

df = df.replace({'False': 0, 'True': 1})  # Handles string booleans

bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

# normalize age col
df['age'] = (df['age'] - df['age'].mean()) / df['age'].std()

# Step 6: Save
df.to_csv('../../train_expanded.csv', index=False)
print("Expanded CSV saved as train_expanded.csv")

# 30 columns