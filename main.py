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
import sys

# import pandas as pd
# from sklearn.model_selection import train_test_split

def main():
    """Main function to run training."""
    # Configuration
    config = {
        'model_name': 'efficientnet_b0',  # or 'vit'  # or 'efficientnet_b0' or 'resnet50' or 'densenet121' or 'convnext_tiny'
        'num_classes': 1,
        'default_metric': ['balanced_accuracy', 'pF1'],          # 'balanced_accuracy' (original) or 'pF1' or 'macroF1' or 'recall' or 'precision'
        'pretrained': True,
        'patched': True,
        'class_weight': False,       # if want to add more weight to positive class (will significantly increase recall and decrease precision)
        'soft_label': False,         # will use BCE loss, and sigmoid. MUST SET num_classes = 1. If set false, MUST SET num_classes = 2
        'soft_pos': 0.9,            # for soft_label. Ignore if not use soft_label
        'soft_neg': 0.0,            # for soft_label. Ignore if not use soft_label
        'threshold': 0.5,           # for soft_label and focal_loss. Ignore if not use soft_label. To increase Precision, Raise the threshold. To increase Recall, Lower the threshold
        'focal_loss': True,         # if True must set soft_label False and num_classes = 1
        'batch_size': 512,          # can be 1024 for H100
        'learning_rate': 1.6e-3,              # set to 1e-4 for small batch_size (linear relationship with batch size)
        'weight_decay': 1.6e-3,               # # set to 1e-4 for small batch_size
        'warmup_start_lr_factor': 0.01,         # Start at 1% of target LR
        'epochs': 40,
        'warmup_epochs': 5,                     # set to 0 for small batch_size (32)
        'patience': 35,
        'num_workers': 6,           # --cpus-per-task=6
        'k_folds': 5,
        'target_col': 'cancer',
        'output_dir': '../outputs'
    }
    
    # Paths (adjust these to your data)
    csv_path = '../../train_expanded.csv'  # Your CSV file with scan_id, patient_id, cancer columns
    data_root = '../../processed_data'      # Root directory containing patient folders with .npy files
    
    # Create trainer
    trainer = BreastCancerTrainer(config)
    
    # Run K-fold cross validation
    fold_results = trainer.train_with_kfold(csv_path, data_root, k_folds=5)
    
    print("Training completed successfully!")

    """ to test the final_test_evaluation

    # Load data
    # df = pd.read_csv(csv_path)
    
    # # Split into train/test first
    # train_df, test_df = train_test_split(
    #     df, 
    #     test_size=0.05, 
    #     stratify=df['cancer'],
    #     random_state=42
    # )

    # # Final evaluation on test set
    # trainer._final_test_evaluation(test_df, data_root)

    """


if __name__ == "__main__":
    # Register the global exception handler
    sys.excepthook = BreastCancerTrainer.handle_exception
    main()