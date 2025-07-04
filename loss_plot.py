import pandas as pd
import matplotlib.pyplot as plt

# Replace with your actual CSV file path
csv_file = '../outputs4-efficientnetb0/fold_2_metrics.csv'

# Read the CSV
df = pd.read_csv(csv_file)

# Extract the relevant columns
epochs = df['Epoch']
# train_loss = df['Train Loss']
# val_loss = df['Val Loss']

# train_loss = df['Train Balanced Acc']
# val_loss = df['Val Balanced Acc']

train_loss = df['Train MacroF1']
val_loss = df['Val MacroF1']

# Plot the losses
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation bal_acc (efficientnetb0-fold2)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
