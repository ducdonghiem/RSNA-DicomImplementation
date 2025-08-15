import pandas as pd

# Load the original dataset
df = pd.read_csv('../train_breast.csv')  # Replace with your actual filename

# Separate positive and negative samples
positive_samples = df[df['cancer'] == 1]
negative_samples = df[df['cancer'] == 0]

# Randomly sample 3,842 negative samples
negative_sampled = negative_samples.sample(n=20000 - 149 - len(positive_samples), random_state=42)

# Combine positive and sampled negative samples
undersampled_df = pd.concat([positive_samples, negative_sampled])

# Shuffle the combined dataset
undersampled_df = undersampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
undersampled_df.to_csv('../train_undersampling_breast_19851.csv', index=False)

print("train_undersampling.csv created with", len(undersampled_df), "samples.")