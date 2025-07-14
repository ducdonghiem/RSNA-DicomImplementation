import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
# arr = np.load("../processed_data/5/640805896.npy")
# arr = np.load("../processed_data/5/1633417959.npy")

# arr = np.load("../processed_data/10006/462822612.npy")
# arr = np.load("../processed_data/10011/220375232.npy")
# arr = np.load("../processed_data/10011/1031443799.npy")
# arr = np.load("../processed_data/10006/1874946579.npy")
# arr = np.load("../processed_data/10011/541722628.npy")
# 

# arr = np.load("../processed_data/patient1/388811999.npy")
# arr = np.load("../processed_data/patient1/613462606.npy")
# arr = np.load("../processed_data/patient2/461614796.npy")
# arr = np.load("../processed_data/patient2/530620473.npy")

# arr = np.load("../processed_data/25/822390278.npy")
# arr = np.load("../processed_data/25/1997933901.npy")

# arr = np.load("../../processed_data/48042/599175404.npy")
# arr = np.load("../../processed_data/10226/461614796.npy")
arr = np.load("../../processed_data/10130/388811999.npy")

print(arr.shape, arr.dtype)     # (512, 512), float32

# Plot the image
plt.imshow(arr, cmap='gray')
plt.axis('off')  # Optional: hides the axis
plt.show()
