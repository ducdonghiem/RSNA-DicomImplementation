from data_preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(
    # data_path="../data/input_dcms",
    # data_path="../data/cancer",
    # data_path="../data",
    data_path = "../../RSNABreastCancerDetection/data/rsna-breast-cancer-detection/train_images",
    # output_path="../processed_data", 
    output_path="../../processed_data", 
    resize_to=(512, 512),
    crop=True,
    apply_voilut=True,
    stretch=True
)

# Get dataset info
stats = preprocessor.get_statistics()
print(stats)

# Process all files
preprocessor.process_dcm()