import pydicom
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut

# dicom = pydicom.dcmread("../data/input_dcms/10006/462822612.dcm")
dicom = pydicom.dcmread("../data/input_dcms/10011/220375232.dcm")

# arr = dicom.pixel_array

# print(arr.dtype)

arr = apply_voi_lut(dicom.pixel_array, dicom)     # output float64, but still integer values

if dicom.PhotometricInterpretation == "MONOCHROME1":
    arr = arr.max() - arr

print(arr.dtype)
# print(arr[0:100, :])

row_means = np.mean(arr, axis=1)
# print(row_means[2500:2600])
# print(np.mean(arr, axis=0))

row_stds = np.std(arr, axis=1)
# print(row_stds[0:100])
# print()
threshold = np.percentile(row_stds, 10)  # could also be a small constant like 1.0
# print(threshold)

std = np.std(arr)

row_threshold = np.percentile(row_means, 30.0)
print(row_threshold)

print(std)

h = np.clip(std / 100, 4.0, 10.0)

print(h)

# means = np.mean(arr)

print(np.percentile(arr, 50.0))