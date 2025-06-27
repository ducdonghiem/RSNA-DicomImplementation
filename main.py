# from DataPreprocessor import DataPreprocessor

# # Initialize preprocessor
# preprocessor = DataPreprocessor(
#     # data_path="../data/input_dcms",
#     # data_path="../data/cancer",
#     # data_path="../data",
#     data_path = "../../RSNABreastCancerDetection/data/rsna-breast-cancer-detection/train_images",
    # output_path="../processed_data", 
#     output_path="../../processed_data", 
    # resize_to=(512, 512),
#     crop=True,
#     apply_voilut=True,
#     stretch=True
# )

# # Get dataset info
# stats = preprocessor.get_statistics()
# print(stats)

# # Process all files
# preprocessor.process_dcm()

#=============================================================================

from BreastCancerTrainer import BreastCancerTrainer

def main():
    """Main function to run training."""
    # Configuration
    config = {
        'model_name': 'efficientnet_b0',  # or 'vit'  # or 'efficientnet_b0' or 'resnet50'
        'num_classes': 2,
        'pretrained': True,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'epochs': 30,
        'patience': 10,
        'num_workers': 4,
        'k_folds': 5,
        'target_col': 'cancer',
        'output_dir': '../outputs'
    }
    
    # Paths (adjust these to your data)
    csv_path = '../../train.csv'  # Your CSV file with scan_id, patient_id, cancer columns
    data_root = '../../processed_data'      # Root directory containing patient folders with .npy files
    
    # Create trainer
    trainer = BreastCancerTrainer(config)
    
    # Run K-fold cross validation
    fold_results = trainer.train_with_kfold(csv_path, data_root, k_folds=5)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()