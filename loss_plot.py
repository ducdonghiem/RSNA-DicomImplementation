import pandas as pd
import matplotlib.pyplot as plt

# Replace with your actual CSV file path
csv_file = '../outputs2-vit/fold_5_metrics.csv'

# Read the CSV
df = pd.read_csv(csv_file)

# Extract the relevant columns
epochs = df['Epoch']
train_loss = df['Train Loss']
val_loss = df['Val Loss']

# Plot the losses
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss (vit-fold5)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
