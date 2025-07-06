from typing import Dict, List, Tuple, Optional, Union
import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
import albumentations as A
import albumentations.pytorch as AP
from tqdm import tqdm
from BreastCancerDataset import BreastCancerDataset
from patch_producer import PatchProducer
from ModelFactory import ModelFactory
from MetricsCalculator import MetricsCalculator

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import csv
import cv2

from augs import CustomRandomSizedCropNoResize

class BreastCancerTrainer:
    """Main trainer class for breast cancer detection models."""
    
    def __init__(self, 
                 config: Dict,
                 device: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.config = config
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model
        self.model = ModelFactory.create_model(
            config['model_name'], 
            num_classes=config.get('num_classes', 2),
            pretrained=config.get('pretrained', True)
        )
        self.model.to(self.device)

        # patch modify
        self.patch_producer = None
        if config['patched']:
            self.patch_producer = PatchProducer()
            self.patch_producer.to(self.device)
        
        # Initialize transforms
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        self.logger.info(f"Trainer initialized with model: {config['model_name']}")
        self.logger.info(f"Device: {self.device}")
    
    # def _setup_logging(self):
    #     """Setup logging configuration."""
    #     logging.basicConfig(
    #         level=logging.INFO,
    #         format='%(asctime)s - %(levelname)s - %(message)s'
    #     )
    #     self.logger = logging.getLogger(__name__)

    def _setup_logging(self):
        """Setup logging configuration to output to both console and file."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # prevent duplicate logs if root logger is configured elsewhere

        # Define log format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        # File handler
        fh = logging.FileHandler(f'{self.config['output_dir']}/training_log.txt', mode='a')  # append to file
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # Clear old handlers (prevent duplicate logs if _setup_logging is called multiple times)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Add handlers
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    @staticmethod
    def handle_exception(exc_type, exc_value, exc_traceback):
        logging.getLogger(__name__).error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    # def _get_train_transforms(self) -> A.Compose:
    #     """Get training transforms."""
    #     return A.Compose([
    #         A.Resize(256, 256),
    #         A.RandomCrop(224, 224),
    #         A.HorizontalFlip(p=0.5),
    #         # A.VerticalFlip(p=0.5),
    #         # A.RandomBrightnessContrast(p=0.2, brightness_limit=0.1, contrast_limit=0.1),
    #         A.OneOf([
    #             A.Blur(blur_limit=3, p=0.5),
    #             # A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    #             # A.GaussNoise(std_range=(0.2, 0.44), mean_range=(0, 0), p=0.5),
    #             A.GaussNoise(std_range=(0.02, 0.08), mean_range=(0.0, 0.0), p=0.5)
    #         ], p=0.2),
    #         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imagenet stats
    #         AP.ToTensorV2()
    #     ])

    # from https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449
    def _get_train_transforms(self) -> A.Compose:
        """Get training transforms."""
        return A.Compose([
            # crop, tweak from A.RandomSizedCrop()
            CustomRandomSizedCropNoResize(scale=(0.8, 1.0), ratio=(0.6, 0.9), p=0.4),       # Original was scale=(0.5, 1.0),  ratio=(0.5, 0.8)
            # flip
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # downscale - FIXED with correct parameters
            A.OneOf([
                A.Downscale(scale_range=(0.75, 0.95), interpolation_pair={"upscale": cv2.INTER_LINEAR, "downscale": cv2.INTER_AREA}, p=0.1),
                A.Downscale(scale_range=(0.75, 0.95), interpolation_pair={"upscale": cv2.INTER_LANCZOS4, "downscale": cv2.INTER_AREA}, p=0.1),
                A.Downscale(scale_range=(0.75, 0.95), interpolation_pair={"upscale": cv2.INTER_LINEAR, "downscale": cv2.INTER_LINEAR}, p=0.8),
            ], p=0.125),
            # contrast
            A.OneOf([
                A.RandomToneCurve(scale=0.3, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.4, 0.5), brightness_by_max=True, p=0.5)
            ], p=0.5),
            # geometric - FIXED with correct parameters
            A.OneOf([
                A.ShiftScaleRotate(
                    shift_limit=0.0, 
                    scale_limit=[-0.15, 0.15], 
                    rotate_limit=[-30, 30], 
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT, 
                    shift_limit_x=[-0.1, 0.1],
                    shift_limit_y=[-0.2, 0.2], 
                    rotate_method='largest_box',
                    fill=0,  # Changed from 'value' to 'fill'
                    fill_mask=0,  # Changed from 'mask_value' to 'fill_mask'
                    p=0.6
                ),
                A.ElasticTransform(
                    alpha=1, 
                    sigma=20, 
                    interpolation=cv2.INTER_LINEAR, 
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,  # Changed from 'value' to 'fill'
                    fill_mask=0,  # Changed from 'mask_value' to 'fill_mask'
                    approximate=False, 
                    same_dxdy=False, 
                    p=0.2
                ),
                A.GridDistortion(
                    num_steps=5, 
                    distort_limit=0.3, 
                    interpolation=cv2.INTER_LINEAR, 
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,  # Changed from 'value' to 'fill'
                    fill_mask=0,  # Changed from 'mask_value' to 'fill_mask'
                    normalized=True, 
                    p=0.2
                ),
            ], p=0.5),
            # random erase - FIXED with correct parameters
            A.CoarseDropout(
                num_holes_range=(1, 6),  # Changed from min_holes/max_holes
                hole_height_range=(0.05, 0.15),  # Changed from min_height/max_height
                hole_width_range=(0.1, 0.25),  # Changed from min_width/max_width
                fill=0,  # Changed from fill_value
                fill_mask=None,  # Changed from mask_fill_value
                p=0.25
            ),
            # ADD THESE MISSING LINES:
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AP.ToTensorV2()
        ], p=0.9)

    # no need to change ?
    def _get_val_transforms(self) -> A.Compose:
        """Get validation transforms."""
        return A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AP.ToTensorV2()
        ])
    
    # useless for now
    def prepare_data(self, csv_path: str, data_root: str, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare train and test data loaders.
        
        Args:
            csv_path: Path to CSV file
            data_root: Root directory for data
            test_size: Fraction of data to use for testing
            
        Returns:
            train_loader, test_loader
        """
        # Load and split data
        df = pd.read_csv(csv_path)
        
        # Stratified split to maintain class balance
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df[self.config.get('target_col', 'cancer')],
            random_state=42     # like set.seed
        )
        
        # Save splits for reproducibility
        os.makedirs(self.config.get('output_dir', 'outputs'), exist_ok=True)
        train_df.to_csv(os.path.join(self.config.get('output_dir', 'outputs'), 'train_split.csv'), index=False)
        test_df.to_csv(os.path.join(self.config.get('output_dir', 'outputs'), 'test_split.csv'), index=False)
        
        # Create datasets
        train_dataset = BreastCancerDataset(
            csv_path=os.path.join(self.config.get('output_dir', 'outputs'), 'train_split.csv'),
            data_root=data_root,
            transform=self.train_transform,
            target_col=self.config.get('target_col', 'cancer')
        )
        
        test_dataset = BreastCancerDataset(
            csv_path=os.path.join(self.config.get('output_dir', 'outputs'), 'test_split.csv'),
            data_root=data_root,
            transform=self.val_transform,
            target_col=self.config.get('target_col', 'cancer')
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if self.device == 'cuda' else False
        )
        
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Test samples: {len(test_dataset)}")
        
        return train_loader, test_loader
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module, 
                   epoch: int) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        model.train()

        # patch modify
        if self.patch_producer:
            self.patch_producer.train()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
        for batch_idx, (images, targets, metadata) in enumerate(pbar):
            images, targets, metadata = images.to(self.device), targets.to(self.device), metadata.to(self.device)
            
            optimizer.zero_grad()
            
            # patch modify
            if self.patch_producer:
                patch = self.patch_producer(metadata)
                images[:, :, :16, 208:] = patch

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            # all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        metrics = MetricsCalculator.calculate_metrics(all_targets, all_preds, all_probs)
        
        return avg_loss, metrics
    
    # appends results from a batch (32 scans) to the full list then compute the metrics after the loop (inference one batch at a time to save memory)
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        model.eval()

        # patch modify
        if self.patch_producer:
            self.patch_producer.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
            for batch_idx, (images, targets, metadata) in enumerate(pbar):
                images, targets, metadata = images.to(self.device), targets.to(self.device), metadata.to(self.device)

                # patch modify
                if self.patch_producer:
                    patch = self.patch_producer(metadata)
                    images[:, :, :16, 208:] = patch
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        avg_loss = total_loss / len(val_loader)
        metrics = MetricsCalculator.calculate_metrics(all_targets, all_preds, all_probs)
        
        return avg_loss, metrics
    
    def train_with_kfold(self, csv_path: str, data_root: str, k_folds: int = 5) -> Dict[str, List[float]]:
        """
        Train model using K-fold cross validation.
        
        Args:
            csv_path: Path to CSV file
            data_root: Root directory for data
            k_folds: Number of folds for cross validation
            
        Returns:
            Dictionary with metrics for each fold
        """
        # Load data
        df = pd.read_csv(csv_path)
        
        # Split into train/test first
        train_df, test_df = train_test_split(
            df, 
            test_size=0.05, 
            stratify=df[self.config.get('target_col', 'cancer')],
            random_state=42
        )
        
        # K-fold cross validation on training data
        # kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        # fold_results['fold'].append(fold + 1)
        #     fold_results['val_accuracy'].append(best_val_metrics['accuracy'])
        #     fold_results['val_balanced_accuracy'].append(best_val_metrics['balanced_accuracy'])
        #     fold_results['val_pF1'].append(best_val_metrics['pF1'])
        #     fold_results['val_macroF1'].append(best_val_metrics['macroF1'])
        #     fold_results['val_auc_roc'].append(best_val_metrics.get('auc_roc', 0.0))
        #     fold_results['val_recall'].append(best_val_metrics['recall'])
        #     fold_results['val_precision'].append(best_val_metrics['precision'])

        fold_results = {
            'fold': [],
            'val_accuracy': [],
            'val_balanced_accuracy': [],
            'val_pF1': [],
            'val_macroF1': [],
            'val_auc_roc': [],
            'val_recall': [],
            'val_precision': []
        }
        
        # for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df)):
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df, train_df[self.config['target_col']])):        # for balance between folds

            self.logger.info(f"Starting Fold {fold + 1}/{k_folds}")
            
            # Create fold datasets
            fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
            
            # Save fold splits
            fold_train_path = os.path.join(self.config.get('output_dir', 'outputs'), f'fold_{fold+1}_train.csv')
            fold_val_path = os.path.join(self.config.get('output_dir', 'outputs'), f'fold_{fold+1}_val.csv')
            fold_train_df.to_csv(fold_train_path, index=False)
            fold_val_df.to_csv(fold_val_path, index=False)
            
            # Create datasets and loaders
            fold_train_dataset = BreastCancerDataset(fold_train_path, data_root, self.train_transform)
            fold_val_dataset = BreastCancerDataset(fold_val_path, data_root, self.val_transform)
            
            fold_train_loader = DataLoader(
                fold_train_dataset, 
                batch_size=self.config.get('batch_size', 32),
                shuffle=True, 
                num_workers=self.config.get('num_workers', 4)
            )
            fold_val_loader = DataLoader(
                fold_val_dataset, 
                batch_size=self.config.get('batch_size', 32),
                shuffle=False, 
                num_workers=self.config.get('num_workers', 4)
            )
            
            # Reinitialize model for each fold
            self.model = ModelFactory.create_model(
                self.config['model_name'], 
                num_classes=self.config.get('num_classes', 2),
                pretrained=self.config.get('pretrained', True)
            )
            self.model.to(self.device)
            
            # Train fold
            best_val_metrics = self._train_fold(fold_train_loader, fold_val_loader, fold + 1)
            
            # Store results
            fold_results['fold'].append(fold + 1)
            fold_results['val_accuracy'].append(best_val_metrics['accuracy'])
            fold_results['val_balanced_accuracy'].append(best_val_metrics['balanced_accuracy'])
            fold_results['val_pF1'].append(best_val_metrics['pF1'])
            fold_results['val_macroF1'].append(best_val_metrics['macroF1'])
            fold_results['val_auc_roc'].append(best_val_metrics.get('auc_roc', 0.0))
            fold_results['val_recall'].append(best_val_metrics['recall'])
            fold_results['val_precision'].append(best_val_metrics['precision'])
        
        # Print fold results
        self._print_fold_results(fold_results)
        
        # Final evaluation on test set
        self._final_test_evaluation(test_df, data_root)
        
        return fold_results
    
    def _train_fold(self, train_loader: DataLoader, val_loader: DataLoader, fold: int) -> Dict[str, float]:
        """Train a single fold."""
        # === 1. Compute class weights ===
        targets = [label for _, label, _ in train_loader.dataset]  # Assumes __getitem__ returns (image, label, meta)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=targets)
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        # Setup optimizer and scheduler
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5,
            # verbose=True
        )
        
        # Setup loss function
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)   #  handle class imbalance by using class weights in the loss function to penalize the model more for misclassifying the minority class (cancer).
        
        # Training loop
        best_val_metrics = None
        best_val_score = 0
        patience_counter = 0
        max_patience = self.config.get('patience', 10)
        
        for epoch in range(self.config.get('epochs', 50)):
            # Train
            train_loss, train_metrics = self.train_epoch(
                self.model, train_loader, optimizer, criterion, epoch
            )
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(
                self.model, val_loader, criterion, epoch
            )
            
            # Update scheduler
            scheduler.step(val_metrics['balanced_accuracy'])
            
            # Check for improvement
            val_score = val_metrics['balanced_accuracy']
            if val_score > best_val_score:
                best_val_score = val_score
                best_val_metrics = val_metrics.copy()
                patience_counter = 0
                
                # Save best model
                model_path = os.path.join(
                    self.config.get('output_dir', 'outputs'), 
                    f'best_model_fold_{fold}.pth'
                )
                torch.save(self.model.state_dict(), model_path)
            else:
                patience_counter += 1
            
            # Log progress
            self.logger.info(
                f'Fold {fold} Epoch {epoch+1}: '
                f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                f'Train Acc: {train_metrics["accuracy"]:.4f}, '
                f'Train Balanced Acc: {train_metrics["balanced_accuracy"]:.4f}, '
                f'Train pF1: {train_metrics["pF1"]:.4f}, '
                f'Train MacroF1: {train_metrics["macroF1"]:.4f}, '
                f'Train AUC: {train_metrics.get('auc_roc', 0.0):.4f}, '
                f'Train Recall: {train_metrics["recall"]:.4f}, '
                f'Train Precision: {train_metrics["precision"]:.4f}, '
                f'Val Acc: {val_metrics["accuracy"]:.4f}, '
                f'Val Balanced Acc: {val_metrics["balanced_accuracy"]:.4f}, '
                f'Val pF1: {val_metrics["pF1"]:.4f}, '
                f'Val MacroF1: {val_metrics["macroF1"]:.4f}, '
                f'Val AUC: {val_metrics.get('auc_roc', 0.0):.4f}, '
                f'Val Recall: {val_metrics["recall"]:.4f}, '
                f'Val Precision: {val_metrics["precision"]:.4f}'
            )

            # CSV file path for the current fold
            csv_file = f"{self.config['output_dir']}/fold_{fold}_metrics.csv"

            # If this is the first epoch, create the file and write the header
            if epoch == 0 and not os.path.exists(csv_file):
                with open(csv_file, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Epoch", "Train Loss", "Val Loss",
                        "Train Acc", "Train Balanced Acc", "Train pF1", "Train MacroF1", "Train AUC", "Train Recall", "Train Precision",
                        "Val Acc", "Val Balanced Acc", "Val pF1", "Val MacroF1", "Val AUC", "Val Recall", "Val Precision"
                    ])

            # Append metrics for the current epoch
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, train_loss, val_loss,
                    train_metrics["accuracy"],
                    train_metrics["balanced_accuracy"],
                    train_metrics["pF1"],
                    train_metrics["macroF1"],
                    train_metrics.get("auc_roc", 0.0),
                    train_metrics["recall"],
                    train_metrics["precision"],
                    val_metrics["accuracy"],
                    val_metrics["balanced_accuracy"],
                    val_metrics["pF1"],
                    val_metrics["macroF1"],
                    val_metrics.get("auc_roc", 0.0),
                    val_metrics["recall"],
                    val_metrics["precision"]
                ])
            
            # Early stopping
            if patience_counter >= max_patience:
                self.logger.info(f'Early stopping triggered at epoch {epoch+1}')
                break
        
        return best_val_metrics
    
    def _print_fold_results(self, fold_results: Dict[str, List[float]]):
        """Print summary of fold results."""
        self.logger.info("\n" + "="*50)
        self.logger.info("K-FOLD CROSS VALIDATION RESULTS")
        self.logger.info("="*50)
        
        for i, fold in enumerate(fold_results['fold']):
            self.logger.info(
                f"Fold {fold}: "
                f"Acc: {fold_results['val_accuracy'][i]:.4f}, "
                f"Balanced Acc: {fold_results['val_balanced_accuracy'][i]:.4f}, "
                f"pF1: {fold_results['val_pF1'][i]:.4f}, "
                f"MacroF1: {fold_results['val_macroF1'][i]:.4f}, "
                f"AUC-ROC: {fold_results['val_auc_roc'][i]:.4f}, "
                f"Recall: {fold_results['val_recall'][i]:.4f}, "
                f"Precision: {fold_results['val_precision'][i]:.4f}"
            )
        
        # Calculate means and stds
        mean_acc = np.mean(fold_results['val_accuracy'])
        std_acc = np.std(fold_results['val_accuracy'])
        mean_bal_acc = np.mean(fold_results['val_balanced_accuracy'])
        std_bal_acc = np.std(fold_results['val_balanced_accuracy'])
        mean_pf1 = np.mean(fold_results['val_pF1'])
        std_pf1 = np.std(fold_results['val_pF1'])
        mean_macro_f1 = np.mean(fold_results['val_macroF1'])
        std_macro_f1 = np.std(fold_results['val_macroF1'])
        mean_auc = np.mean(fold_results['val_auc_roc'])
        std_auc = np.std(fold_results['val_auc_roc'])
        mean_recall = np.mean(fold_results['val_recall'])
        std_recall = np.std(fold_results['val_recall'])
        mean_precision = np.mean(fold_results['val_precision'])
        std_precision = np.std(fold_results['val_precision'])
        
        self.logger.info("-" * 50)

        self.logger.info(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        self.logger.info(f"Mean Balanced Accuracy: {mean_bal_acc:.4f} ± {std_bal_acc:.4f}")
        self.logger.info(f"Mean pF1 Score: {mean_pf1:.4f} ± {std_pf1:.4f}")
        self.logger.info(f"Mean macroF1 Score: {mean_macro_f1:.4f} ± {std_macro_f1:.4f}")
        self.logger.info(f"Mean AUC-ROC: {mean_auc:.4f} ± {std_auc:.4f}")
        self.logger.info(f"Mean Recall: {mean_recall:.4f} ± {std_recall:.4f}")
        self.logger.info(f"Mean Precision: {mean_precision:.4f} ± {std_precision:.4f}")
        self.logger.info("="*50)
    
    # only test model 5
    # def _final_test_evaluation(self, test_df: pd.DataFrame, data_root: str):
    #     """Evaluate on final test set."""
    #     self.logger.info("\nFinal Test Set Evaluation")
    #     self.logger.info("-" * 30)
        
    #     # Save test split
    #     test_path = os.path.join(self.config.get('output_dir', 'outputs'), 'final_test_split.csv')
    #     test_df.to_csv(test_path, index=False)
        
    #     # Create test dataset
    #     test_dataset = BreastCancerDataset(test_path, data_root, self.val_transform)
    #     test_loader = DataLoader(
    #         test_dataset,
    #         batch_size=self.config.get('batch_size', 32),
    #         shuffle=False, 
    #         num_workers=self.config.get('num_workers', 4)
    #     )
        
    #     # Load best model from last fold (or you could ensemble)
    #     best_model_path = os.path.join(
    #         self.config.get('output_dir', 'outputs'), 
    #         f'best_model_fold_{self.config.get("k_folds", 5)}.pth'
    #     )
        
    #     if os.path.exists(best_model_path):
    #         self.model.load_state_dict(torch.load(best_model_path))
    #         self.logger.info(f"Loaded best model from {best_model_path}")
        
    #     # Evaluate
    #     # # === 1. Compute class weights === Dont apply this for test evaluation (chatGPT said so)
    #     # targets = [label for _, label, _ in test_loader.dataset]  # Assumes __getitem__ returns (image, label, meta)
    #     # class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=targets)
    #     # weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

    #     criterion = nn.CrossEntropyLoss()

    #     # use this function to obtain results for the test set (only 1 epoch)
    #     test_loss, test_metrics = self.validate_epoch(self.model, test_loader, criterion, 0)
        
    #     self.logger.info(f"Test Loss: {test_loss:.4f}")
    #     self.logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    #     self.logger.info(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    #     self.logger.info(f"Test pF1: {test_metrics['pF1']:.4f}")
    #     self.logger.info(f"Test macroF1: {test_metrics['macroF1']:.4f}")
    #     self.logger.info(f"Test AUC-ROC: {test_metrics.get('auc_roc', 0.0):.4f}")
    #     self.logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    #     self.logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        
    #     # Save test results
    #     results = {
    #         'test_loss': test_loss,
    #         'test_metrics': test_metrics,
    #         'config': self.config
    #     }
        
    #     results_path = os.path.join(self.config.get('output_dir', 'outputs'), 'final_test_results.json')
    #     with open(results_path, 'w') as f:
    #         json.dump(results, f, indent=2)
        
    #     self.logger.info(f"Results saved to {results_path}")

    # ensemble all 5 models
    def _final_test_evaluation(self, test_df: pd.DataFrame, data_root: str):
        """Evaluate on final test set using ensemble of all fold models."""
        self.logger.info("\nFinal Test Set Evaluation")
        self.logger.info("-" * 30)
        
        # Save test split
        test_path = os.path.join(self.config.get('output_dir', 'outputs'), 'final_test_split.csv')
        test_df.to_csv(test_path, index=False)
        
        # Create test dataset
        test_dataset = BreastCancerDataset(test_path, data_root, self.val_transform)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.get('batch_size', 32),
            shuffle=False, 
            num_workers=self.config.get('num_workers', 4)
        )
        
        # Load all fold models
        models = []
        k_folds = self.config.get("k_folds", 5)
        for fold in range(1, k_folds + 1):
            model_path = os.path.join(
                self.config.get('output_dir', 'outputs'), 
                f'best_model_fold_{fold}.pth'
            )
            if os.path.exists(model_path):
                # fold_model = type(self.model)()  # Create new instance
                fold_model = ModelFactory.create_model(
                    self.config['model_name'],
                    num_classes=self.config.get('num_classes', 2),
                    pretrained=False
                )
                fold_model.load_state_dict(torch.load(model_path))
                fold_model.to(self.device)
                fold_model.eval()

                # patch modify
                if self.patch_producer:
                    self.patch_producer.eval()

                models.append(fold_model)
                self.logger.info(f"Loaded model from fold {fold}")
        
        # Ensemble prediction
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Test [Ensemble]')
            for batch_idx, (images, targets, metadata) in enumerate(pbar):
                images, targets, metadata = images.to(self.device), targets.to(self.device), metadata.to(self.device)

                # patch modify
                if self.patch_producer:
                    patch = self.patch_producer(metadata)
                    images[:, :, :16, 208:] = patch
                
                # Average predictions from all models
                ensemble_outputs = torch.zeros_like(models[0](images))
                for model in models:
                    ensemble_outputs += model(images)
                ensemble_outputs /= len(models)
                
                loss = criterion(ensemble_outputs, targets)
                total_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(ensemble_outputs, dim=1)
                preds = torch.argmax(ensemble_outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        test_loss = total_loss / len(test_loader)
        test_metrics = MetricsCalculator.calculate_metrics(all_targets, all_preds, all_probs)
        
        self.logger.info(f"Test Loss: {test_loss:.4f}")
        self.logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        self.logger.info(f"Test pF1: {test_metrics['pF1']:.4f}")
        self.logger.info(f"Test macroF1: {test_metrics['macroF1']:.4f}")
        self.logger.info(f"Test AUC-ROC: {test_metrics.get('auc_roc', 0.0):.4f}")
        self.logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
        self.logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        
        # Save test results
        results = {
            'test_loss': test_loss,
            'test_metrics': test_metrics,
            'config': self.config
        }
        
        results_path = os.path.join(self.config.get('output_dir', 'outputs'), 'final_test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_path}")