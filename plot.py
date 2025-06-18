import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
arr = np.load("../processed_data/5/640805896.npy")

# Plot the image
plt.imshow(arr, cmap='gray')
plt.axis('off')  # Optional: hides the axis
plt.show()
