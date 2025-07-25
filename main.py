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
        'default_metric': 'pF1', # ['balanced_accuracy', 'pF1'],# 'balanced_accuracy' (original) or 'pF1' or 'macroF1' or 'recall' or 'precision'
        'pretrained': True,
        'patched': True,
        'class_weight': False,       # if want to add more weight to positive class (will significantly increase recall and decrease precision)
        'soft_label': False,         # will use BCE loss, and sigmoid. MUST SET num_classes = 1. If set false, MUST SET num_classes = 2
        'soft_pos': 0.9,            # for soft_label. Ignore if not use soft_label
        'soft_neg': 0.0,            # for soft_label. Ignore if not use soft_label
        'threshold': 0.32,           # for soft_label and focal_loss. Ignore if not use soft_label. To increase Precision, Raise the threshold. To increase Recall, Lower the threshold
        'focal_loss': True,         # if True must set soft_label False and num_classes = 1
        'batch_size': 64,          # can be 1024 for H100
        'learning_rate': 2e-4,              # set to 1e-4 for small batch_size (linear relationship with batch size)
        'weight_decay': 2e-4,               # # set to 1e-4 for small batch_size
        'warmup_start_lr_factor': 0.01,         # Start at 1% of target LR
        'epochs': 60,
        'warmup_epochs': 5,                     # set to 0 for small batch_size (32)
        'patience': 55,
        'num_workers': 12,           # --cpus-per-task=6
        'test_size': 0.1,
        'k_folds': 5,
        'target_col': 'cancer',
        'output_dir': '../outputs',
        'use_external_data': False,
        'oversample_minority': True             # minority oversammpling (no need external data)
    }

    if config['use_external_data']:
        config['external_csv_path'] = '../../external_expanded.csv'
    
    # Paths (adjust these to your data)
    csv_path = '../../train_expanded.csv'  # Your CSV file with scan_id, patient_id, cancer columns
    data_root = '../../processed_data'      # Root directory containing patient folders with .npy files
    data_root_external = '../../processed_external_data'
    
    # Create trainer
    trainer = BreastCancerTrainer(config)
    
    # Run K-fold cross validation
    fold_results = trainer.train_with_kfold(csv_path, data_root, data_root_external, k_folds=config['k_folds'], resume=False)       # set resume=True if use checkpoint
    
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