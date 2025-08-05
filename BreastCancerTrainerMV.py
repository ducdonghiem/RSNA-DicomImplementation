import matplotlib
matplotlib.use('Agg') # This must be called BEFORE importing pyplot
import matplotlib.pyplot as plt

# Calculate Precision-Recall curve points
from sklearn.metrics import precision_recall_curve, balanced_accuracy_score
# Evaluate with the new optimal threshold on the combined validation data (optional, but good for reporting)

import glob
import re # For parsing epoch from filenames

from typing import Dict, List, Tuple, Optional, Union
import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, KFold
import albumentations as A
import albumentations.pytorch as AP
from tqdm import tqdm
# from BreastCancerDataset import BreastCancerDataset
from BreastCancerDatasetMV import BreastCancerDatasetMV
from patch_producer import PatchProducer
from ModelFactory import ModelFactory
from MetricsCalculator import MetricsCalculator

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import csv
import cv2

from augs import CustomRandomSizedCropNoResize

from FocalLoss import FocalLoss

class BreastCancerTrainerMV:
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

         # determine whether to include metadata depending on patch usage
        self.include_metadata = True if self.patch_producer else False

        # if config['soft_label']:
        
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

        # In BreastCancerTrainer class or as a global constant
        self.METRIC_TO_CSV_COL_MAP = {
            'accuracy': 'Acc',
            'balanced_accuracy': 'Balanced Acc',
            'pF1': 'pF1',          # This ensures 'pF1' remains 'pF1'
            'macroF1': 'MacroF1',  # This ensures 'MacroF1' remains 'MacroF1'
            'auc_roc': 'AUC',
            'recall': 'Recall',
            'precision': 'Precision'
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
        fh = logging.FileHandler(f"{self.config['output_dir']}/training_log.txt", mode='a')  # append to file
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
                A.Compose([
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
                    # A.ShiftScaleRotate(
                    #     shift_limit=0.0, 
                    #     scale_limit=[-0.15, 0.15], 
                    #     rotate_limit=[-30, 30], 
                    #     interpolation=cv2.INTER_LINEAR,
                    #     border_mode=cv2.BORDER_CONSTANT, 
                    #     shift_limit_x=[-0.1, 0.1],
                    #     shift_limit_y=[-0.2, 0.2], 
                    #     rotate_method='largest_box',
                    #     fill=0,  # Changed from 'value' to 'fill'
                    #     fill_mask=0,  # Changed from 'mask_value' to 'fill_mask'
                    #     p=0.6
                    # ),
                    A.Affine(
                        scale=(-0.15, 0.15),  # scale_limit=[-0.15, 0.15]
                        translate_percent={
                            'x': (-0.1, 0.1),   # shift_limit_x=[-0.1, 0.1]
                            'y': (-0.2, 0.2)    # shift_limit_y=[-0.2, 0.2]
                        },
                        rotate=(-30, 30),       # rotate_limit=[-30, 30]
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_CONSTANT,
                        rotate_method='largest_box',
                        fill=0,
                        fill_mask=0,
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
            ], p=0.9),
            # ADD THESE MISSING LINES:
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AP.ToTensorV2()
          ])

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
        train_dataset = BreastCancerDatasetMV(
            csv_path=os.path.join(self.config.get('output_dir', 'outputs'), 'train_split.csv'),
            data_root=data_root,
            transform=self.train_transform,
            target_col=self.config.get('target_col', 'cancer'),
            include_metadata=self.include_metadata,
            data_root_external=None
        )
        
        test_dataset = BreastCancerDatasetMV(
            csv_path=os.path.join(self.config.get('output_dir', 'outputs'), 'test_split.csv'),
            data_root=data_root,
            transform=self.val_transform,
            target_col=self.config.get('target_col', 'cancer'),
            include_metadata=self.include_metadata,
            data_root_external=None
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

    def _find_resume_point(self, output_dir: str) -> Tuple[int, int]:
        """
        Scans the output directory to find the last completed fold and epoch.
        Returns (resume_fold_idx, resume_epoch_idx) (0-indexed fold, 0-indexed epoch)
        """
        latest_fold = -1
        latest_epoch_in_fold = -1

        # 1. Check for best_model_fold_X_epoch_Y.pth files
        model_files = glob.glob(os.path.join(output_dir, 'best_model_fold_*_epoch_*.pth'))
        
        for fpath in model_files:
            match = re.search(r'best_model_fold_(\d+)_epoch_(\d+)\.pth$', fpath)
            if match:
                fold_num = int(match.group(1)) - 1 # Convert to 0-indexed fold
                epoch_num = int(match.group(2)) - 1 # Convert to 0-indexed epoch
                
                if fold_num > latest_fold:
                    latest_fold = fold_num
                    latest_epoch_in_fold = epoch_num
                elif fold_num == latest_fold and epoch_num > latest_epoch_in_fold:
                    latest_epoch_in_fold = epoch_num

        if latest_fold != -1:
            self.logger.info(f"Resuming from Fold {latest_fold + 1}, Epoch {latest_epoch_in_fold + 1}")
            return latest_fold, latest_epoch_in_fold + 1 # Return 1-indexed epoch for loop range
        else:
            self.logger.info("No previous checkpoints found. Starting from scratch.")
            return 0, 0 # Start from fold 0, epoch 0
    
    def mv_collate(self, batch):
        cc_list_batch = [sample[0] for sample in batch]  # list of lists
        mlo_list_batch = [sample[1] for sample in batch]
        targets = torch.tensor([sample[2] for sample in batch])
        metadata = [sample[3] for sample in batch]
        breast_ids = [sample[4] for sample in batch]

        return cc_list_batch, mlo_list_batch, targets, metadata, breast_ids
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module, 
                   epoch: int, warmup_steps: int) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        model.train()

        # patch modify
        if self.patch_producer:
            self.patch_producer.train()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        # for batch_idx, (images, targets, metadata, image_ids) in enumerate(pbar):
        for batch_idx, (images_CC, images_MLO, targets, metadata, breast_ids) in enumerate(pbar):
            # images_CC, images_MLO, targets, metadata = images_CC.to(self.device), images_MLO.to(self.device), targets.to(self.device), metadata.to(self.device)
            targets = targets.to(self.device)

            images_CC = [[view.to(self.device) for view in sample] for sample in images_CC]
            images_MLO = [[view.to(self.device) for view in sample] for sample in images_MLO]

            global_step = epoch * len(train_loader) + batch_idx         # current step, or current iteration, or current batch
            # --- Learning Rate Warm-up Logic ---
            if global_step < warmup_steps:
                # Linearly increase LR from warmup_start_lr_factor * target_lr to target_lr
                warmup_factor = global_step / warmup_steps
                lr = self.config['learning_rate'] * (self.config['warmup_start_lr_factor'] + (1 - self.config['warmup_start_lr_factor']) * warmup_factor)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            # --- End Warm-up Logic ---

            optimizer.zero_grad()
            
            # patch modify
            if self.patch_producer:
                patch = self.patch_producer(metadata)  # shape: (B, C, 16, W_patch)
                patch = patch.to(self.device)

                # Apply patch to each CC view
                for i in range(images_CC.shape[1]):  # n_CC views
                    images_CC[:, i, :, :16, 208:] = patch

                # Apply patch to each MLO view
                for i in range(images_MLO.shape[1]):  # n_MLO views
                    images_MLO[:, i, :, :16, 208:] = patch

            # images_CC = [images_CC[:, i] for i in range(images_CC.size(1))]
            # images_MLO = [images_MLO[:, j] for j in range(images_MLO.size(1))]

            if self.config['soft_label']:
                soft_targets = targets.float().clone()
                soft_targets[soft_targets == 1] = self.config['soft_pos']
                soft_targets[soft_targets == 0] = self.config['soft_neg']
                # soft_targets = soft_targets.unsqueeze(1)  # Ensure shape [B, 1] for BCEWithLogitsLoss
                outputs = model(images_MLO, images_CC).squeeze(1)
                loss = criterion(outputs, soft_targets)
            else:
                outputs = model(images_MLO, images_CC)
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            # Print LR for monitoring (optional)
            if global_step % 400 == 0: # Print every 400 steps
                self.logger.info(f"LR MONITOR: Epoch: {epoch}, Step: {global_step}, LR: {optimizer.param_groups[0]['lr']:.6f}, Loss: {loss.item():.4f}")
            
            total_loss += loss.item()
            
            # Get predictions
            if self.config['soft_label'] or self.config['focal_loss']:
                probs = torch.sigmoid(outputs)            # Correct for BCEWithLogitsLoss
                preds = (probs >= self.config['threshold']).long()             # or tune this threshold if needed
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            if self.config['soft_label'] or self.config['focal_loss']:
                all_probs.extend(probs.detach().cpu().numpy())  # probs is already class 1
            else:
                all_probs.extend(probs[:, 1].detach().cpu().numpy())  # take prob for class 1
            
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
                      criterion: nn.Module, epoch: int) -> Tuple[float, Dict[str, float], List[int], List[float]]:
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
            # for batch_idx, (images, targets, metadata, image_ids) in enumerate(pbar):
            for batch_idx, (images_CC, images_MLO, targets, metadata, breast_ids) in enumerate(pbar):
                # images_CC, images_MLO, targets, metadata = images_CC.to(self.device), images_MLO.to(self.device), targets.to(self.device), metadata.to(self.device)
                targets = targets.to(self.device)

                images_CC = [[view.to(self.device) for view in sample] for sample in images_CC]
                images_MLO = [[view.to(self.device) for view in sample] for sample in images_MLO]

                 # patch modify
                if self.patch_producer:
                    patch = self.patch_producer(metadata)  # shape: (B, C, 16, W_patch)
                    patch = patch.to(self.device)

                    # Apply patch to each CC view
                    for i in range(images_CC.shape[1]):  # n_CC views
                        images_CC[:, i, :, :16, 208:] = patch

                    # Apply patch to each MLO view
                    for i in range(images_MLO.shape[1]):  # n_MLO views
                        images_MLO[:, i, :, :16, 208:] = patch

                # images_CC = [images_CC[:, i] for i in range(images_CC.size(1))]
                # images_MLO = [images_MLO[:, j] for j in range(images_MLO.size(1))]

                if self.config['soft_label']:
                    soft_targets = targets.float().clone()
                    soft_targets[soft_targets == 1] = self.config['soft_pos']
                    soft_targets[soft_targets == 0] = self.config['soft_neg']
                    # soft_targets = soft_targets.unsqueeze(1)  # Ensure shape [B, 1] for BCEWithLogitsLoss
                    outputs = model(images_MLO, images_CC).squeeze(1)
                    loss = criterion(outputs, soft_targets)
                else:
                    outputs = model(images_MLO, images_CC)
                    loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                
                # Get predictions
                if self.config['soft_label'] or self.config['focal_loss']:
                    probs = torch.sigmoid(outputs)            # Correct for BCEWithLogitsLoss
                    preds = (probs >= self.config['threshold']).long()             # or tune this threshold if needed
                else:
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                if self.config['soft_label']or self.config['focal_loss']:
                    all_probs.extend(probs.detach().cpu().numpy())  # probs is already class 1
                else:
                    all_probs.extend(probs[:, 1].detach().cpu().numpy())  # take prob for class 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        avg_loss = total_loss / len(val_loader)
        metrics = MetricsCalculator.calculate_metrics(all_targets, all_preds, all_probs)
        
        return avg_loss, metrics, all_targets, all_probs
    
    def train_with_kfold(self, csv_path: str, data_root: str, data_root_external: Optional[str]=None, k_folds: int = 5, resume: bool = False) -> Dict[str, List[float]]:
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
        df_main = pd.read_csv(csv_path)

        use_external = self.config.get('use_external_data', False)
        external_csv_path = self.config.get('external_csv_path') if use_external else None
        df_external = None # Initialize df_external
        if use_external:
            if external_csv_path is None or not os.path.exists(external_csv_path):
                raise FileNotFoundError(
                    f"External dataset requested but path '{external_csv_path}' does not exist."
                )
            df_external = pd.read_csv(external_csv_path)
            self.logger.info(f"Loaded external dataset with {len(df_external)} samples from {external_csv_path}")
        else:
            df_external = None
        
        # Split into train/test first
        train_df_main, test_df = train_test_split(
            df_main, 
            test_size=self.config['test_size'],
            stratify=df_main[self.config.get('target_col', 'cancer')],
            random_state=42
        )

        # if use_external and df_external is not None:
        #     train_df = pd.concat([train_df_main, df_external], ignore_index=True)
        # else:
        #     train_df = train_df_main
        
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

        start_fold_idx = 0
        start_epoch_idx = 0

        if resume:
            start_fold_idx, start_epoch_idx = self._find_resume_point(self.config.get('output_dir', 'outputs'))
            # If resuming a fold, we need to load its previous results into fold_results
            if start_fold_idx > 0: # If not starting from the very first fold
                for f_idx in range(start_fold_idx):
                    csv_file = f"{self.config['output_dir']}/fold_{f_idx+1}_metrics.csv"
                    if os.path.exists(csv_file):
                        df_metrics = pd.read_csv(csv_file)
                        last_row = df_metrics.iloc[-1] # Get metrics from the last epoch of completed folds
                        fold_results['fold'].append(f_idx + 1)
                        fold_results['val_accuracy'].append(last_row['Val Acc'])
                        fold_results['val_balanced_accuracy'].append(last_row['Val Balanced Acc'])
                        fold_results['val_pF1'].append(last_row['Val pF1'])
                        fold_results['val_macroF1'].append(last_row['Val MacroF1'])
                        fold_results['val_auc_roc'].append(last_row['Val AUC'])
                        fold_results['val_recall'].append(last_row['Val Recall'])
                        fold_results['val_precision'].append(last_row['Val Precision']) 
                    else:
                        self.logger.warning(f"Metrics CSV for completed fold {f_idx+1} not found: {csv_file}")

        # Add new lists to store true labels and probabilities across all folds for final tuning
        all_folds_true_labels = []
        all_folds_predicted_probs = []
        
        # for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df)):
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_df_main, train_df_main[self.config['target_col']])):     # for balance between folds
            current_fold_start_epoch = 0
            fold_true_labels = []
            fold_predicted_probs = []

            if resume and fold_idx < start_fold_idx:
                # This fold was completed in a previous run. Load its best model and re-evaluate its validation set.
                self.logger.info(f"Processing completed Fold {fold_idx + 1} for metrics collection...")

                # Load pre-saved fold splits
                fold_val_path = os.path.join(self.config.get('output_dir', 'outputs'), f'fold_{fold_idx+1}_val.csv')
                if not os.path.exists(fold_val_path):
                    self.logger.error(f"Validation CSV for completed fold {fold_idx+1} not found at {fold_val_path}. Cannot collect metrics.")
                    continue # Skip this fold if data is missing
                
                # fold_val_df = pd.read_csv(fold_val_path)
                # Corrected: Pass df as keyword argument
                fold_val_dataset = BreastCancerDatasetMV(csv_path=fold_val_path, data_root=data_root, transform=self.val_transform, target_col=self.config.get('target_col', 'cancer'), include_metadata=self.include_metadata, data_root_external=data_root_external)
                
                fold_val_loader = DataLoader(
                    fold_val_dataset, 
                    batch_size=self.config.get('batch_size', 32),
                    shuffle=False, 
                    num_workers=self.config.get('num_workers', 4),
                    collate_fn=self.mv_collate
                )

                # Load the best model for this completed fold
                model_for_eval = ModelFactory.create_model(
                    self.config['model_name'], 
                    num_classes=self.config.get('num_classes', 1),
                    pretrained=self.config.get('pretrained', True)
                )
                
                # Find the best model for this fold (highest epoch)
                search_pattern = os.path.join(self.config.get('output_dir', 'outputs'), f'best_model_fold_{fold_idx+1}_epoch_*.pth')
                found_models = glob.glob(search_pattern)
                
                if not found_models:
                    self.logger.warning(f"No best model found for completed fold {fold_idx+1} at '{search_pattern}'. Skipping metrics collection for this fold.")
                    continue
                
                # Assume the highest epoch model is the best one for a completed fold
                latest_model_path = None
                max_epoch = -1
                for model_file in found_models:
                    match = re.search(r'_epoch_(\d+)\.pth$', model_file)
                    if match:
                        epoch_num = int(match.group(1))
                        if epoch_num > max_epoch:
                            max_epoch = epoch_num
                            latest_model_path = model_file
                
                if latest_model_path:
                    self.logger.info(f"Loading best model for completed Fold {fold_idx+1} from {latest_model_path}")
                    model_for_eval.load_state_dict(torch.load(latest_model_path, map_location=self.device))
                    model_for_eval.to(self.device)
                    # model_for_eval.eval()

                    # Re-run validation inference to get true labels and probabilities
                    temp_crit = None
                    if self.config['focal_loss']:
                        temp_crit = FocalLoss()
                    elif self.config['soft_label']:
                        temp_crit = nn.BCEWithLogitsLoss()
                    else:
                        temp_crit = nn.CrossEntropyLoss()
                    _, _, fold_true_labels, fold_predicted_probs = self.validate_epoch(
                        model_for_eval, fold_val_loader, temp_crit, 0 # Criterion and epoch don't matter for just getting predictions
                    )
                else:
                    self.logger.warning(f"Could not load best model for completed Fold {fold_idx+1}. Skipping metrics collection.")
                    continue

            elif resume and fold_idx == start_fold_idx:                          
                # This is the fold to resume training for
                current_fold_start_epoch = start_epoch_idx
                self.logger.info(f"Resuming Fold {fold_idx + 1} from Epoch {current_fold_start_epoch + 1}")
                
                # Load pre-saved fold splits for resuming
                fold_train_path = os.path.join(self.config.get('output_dir', 'outputs'), f'fold_{fold_idx+1}_train.csv')
                fold_val_path = os.path.join(self.config.get('output_dir', 'outputs'), f'fold_{fold_idx+1}_val.csv')
                
                if not os.path.exists(fold_train_path) or not os.path.exists(fold_val_path):
                    self.logger.error(f"Resume failed: Training/Validation CSVs for fold {fold_idx+1} not found.")
                    raise FileNotFoundError(f"Missing resume data for fold {fold_idx+1}")
                
                # fold_train_df = pd.read_csv(fold_train_path)
                # fold_val_df = pd.read_csv(fold_val_path)

                # Corrected: Pass df as keyword argument
                fold_train_dataset = BreastCancerDatasetMV(csv_path=fold_train_path, data_root=data_root, transform=self.train_transform, target_col=self.config.get('target_col', 'cancer'), include_metadata=self.include_metadata, data_root_external=data_root_external)
                fold_val_dataset = BreastCancerDatasetMV(csv_path=fold_val_path, data_root=data_root, transform=self.val_transform, target_col=self.config.get('target_col', 'cancer'), include_metadata=self.include_metadata, data_root_external=data_root_external)
                
                if self.config['oversample_minority']:
                    # --- WeightedRandomSampler for minority oversampling ---
                    # Get targets for the current training fold (only from main data, external data is not used for weighting)
                    fold_train_df = pd.read_csv(fold_train_path)
                    train_targets = fold_train_df[self.config.get('target_col', 'cancer')].values
                    # Calculate class weights for sampling
                    class_sample_counts = np.array([len(np.where(train_targets == t)[0]) for t in np.unique(train_targets)])
                    weight = 1. / class_sample_counts
                    samples_weight = np.array([weight[t] for t in train_targets])
                    samples_weight = torch.from_numpy(samples_weight).double()
                    # Create sampler
                    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
                    fold_train_loader = DataLoader(
                        fold_train_dataset, 
                        batch_size=self.config.get('batch_size', 32),
                        # shuffle=True, # Remove shuffle when using a sampler
                        sampler=sampler, # Use the sampler here
                        num_workers=self.config.get('num_workers', 4),
                        collate_fn=self.mv_collate
                    )
                else:
                    fold_train_loader = DataLoader(
                        fold_train_dataset, 
                        batch_size=self.config.get('batch_size', 32),
                        shuffle=True, 
                        num_workers=self.config.get('num_workers', 4),
                        collate_fn=self.mv_collate
                    )

                fold_val_loader = DataLoader(
                    fold_val_dataset, 
                    batch_size=self.config.get('batch_size', 32),
                    shuffle=False, 
                    num_workers=self.config.get('num_workers', 4),
                    collate_fn=self.mv_collate
                )
                
                # Reinitialize model for each fold (or load checkpoint if resuming)
                self.model = ModelFactory.create_model(
                    self.config['model_name'], 
                    num_classes=self.config.get('num_classes', 1), # Changed to 1
                    pretrained=self.config.get('pretrained', True)
                )
                self.model.to(self.device)
                
                # Train fold
                best_val_metrics, fold_true_labels, fold_predicted_probs = self._train_fold(
                    fold_train_loader, fold_val_loader, fold_idx + 1, start_epoch=current_fold_start_epoch
                )

            else: # This block handles both fresh runs (resume=False) and new folds after a resume point
                self.logger.info(f"Starting Fold {fold_idx + 1}/{k_folds}")
                # For new folds, generate splits as usual
                fold_train_df_main_subset = train_df_main.iloc[train_idx].reset_index(drop=True)
                fold_val_df = train_df_main.iloc[val_idx].reset_index(drop=True) # Validation is always main data

                # Construct the full training DataFrame for this fold (main subset + external if applicable)
                if use_external and df_external is not None:
                    fold_train_df = pd.concat([fold_train_df_main_subset, df_external], ignore_index=True)
                else:
                    fold_train_df = fold_train_df_main_subset
                
                # Save fold splits (only for new folds, or if not resuming this specific fold)
                fold_train_path = os.path.join(self.config.get('output_dir', 'outputs'), f'fold_{fold_idx+1}_train.csv')
                fold_val_path = os.path.join(self.config.get('output_dir', 'outputs'), f'fold_{fold_idx+1}_val.csv')
                fold_train_df.to_csv(fold_train_path, index=False)
                fold_val_df.to_csv(fold_val_path, index=False)
            
                # Corrected: Pass csv_path as keyword argument
                fold_train_dataset = BreastCancerDatasetMV(csv_path=fold_train_path, data_root=data_root, transform=self.train_transform, target_col=self.config.get('target_col', 'cancer'), include_metadata=self.include_metadata, data_root_external=data_root_external)
                fold_val_dataset = BreastCancerDatasetMV(csv_path=fold_val_path, data_root=data_root, transform=self.val_transform, target_col=self.config.get('target_col', 'cancer'), include_metadata=self.include_metadata, data_root_external=data_root_external)
                
                if self.config['oversample_minority']:
                    # --- WeightedRandomSampler for minority oversampling ---
                    # Get targets for the current training fold (only from main data, external data is not used for weighting)
                    train_targets = fold_train_df[self.config.get('target_col', 'cancer')].values
                    # Calculate class weights for sampling
                    class_sample_counts = np.array([len(np.where(train_targets == t)[0]) for t in np.unique(train_targets)])
                    weight = 1. / class_sample_counts
                    samples_weight = np.array([weight[t] for t in train_targets])
                    samples_weight = torch.from_numpy(samples_weight).double()
                    # Create sampler
                    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
                    fold_train_loader = DataLoader(
                        fold_train_dataset, 
                        batch_size=self.config.get('batch_size', 32),
                        # shuffle=True, # Remove shuffle when using a sampler
                        sampler=sampler, # Use the sampler here
                        num_workers=self.config.get('num_workers', 4),
                        collate_fn=self.mv_collate
                    )
                else:
                    fold_train_loader = DataLoader(
                        fold_train_dataset, 
                        batch_size=self.config.get('batch_size', 32),
                        shuffle=True, 
                        num_workers=self.config.get('num_workers', 4),
                        collate_fn=self.mv_collate
                    )

                fold_val_loader = DataLoader(
                    fold_val_dataset, 
                    batch_size=self.config.get('batch_size', 32),
                    shuffle=False, 
                    num_workers=self.config.get('num_workers', 4),
                    collate_fn=self.mv_collate
                )
                
                # Reinitialize model for each fold (or load checkpoint if resuming)
                self.model = ModelFactory.create_model(
                    self.config['model_name'], 
                    num_classes=self.config.get('num_classes', 1), # Changed to 1
                    pretrained=self.config.get('pretrained', True)
                )
                self.model.to(self.device)
                
                # Train fold
                best_val_metrics, fold_true_labels, fold_predicted_probs = self._train_fold(
                    fold_train_loader, fold_val_loader, fold_idx + 1, start_epoch=current_fold_start_epoch
                )

            if not (resume and fold_idx < start_fold_idx) and best_val_metrics is not None:
                # Store results
                fold_results['fold'].append(fold_idx + 1)
                fold_results['val_accuracy'].append(best_val_metrics['accuracy'])
                fold_results['val_balanced_accuracy'].append(best_val_metrics['balanced_accuracy'])
                fold_results['val_pF1'].append(best_val_metrics['pF1'])
                fold_results['val_macroF1'].append(best_val_metrics['macroF1'])
                fold_results['val_auc_roc'].append(best_val_metrics.get('auc_roc', 0.0))
                fold_results['val_recall'].append(best_val_metrics['recall'])
                fold_results['val_precision'].append(best_val_metrics['precision'])

            # --- Store true labels and probabilities for this fold ---
            all_folds_true_labels.append(fold_true_labels)
            all_folds_predicted_probs.append(fold_predicted_probs)
        
        # Print fold results
        self._print_fold_results(fold_results)

        # --- POST-TRAINING THRESHOLD TUNING AND PR CURVE PLOTTING ---
        self.logger.info("Performing post-training threshold tuning and PR curve analysis...")

        # Concatenate all true labels and probabilities from validation sets across folds
        # This gives you a large, representative dataset for tuning
        combined_true_labels = [label for sublist in all_folds_true_labels for label in sublist]
        combined_predicted_probs = [prob for sublist in all_folds_predicted_probs for prob in sublist]

        precisions, recalls, thresholds = precision_recall_curve(combined_true_labels, combined_predicted_probs)

        # Find the threshold that maximizes a desired metric (e.g., F1-score)
        # (You can choose to maximize something else, like balanced accuracy or recall at a certain precision)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10) # Add epsilon to avoid division by zero
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1_score = f1_scores[optimal_idx]
        optimal_precision = precisions[optimal_idx]
        optimal_recall = recalls[optimal_idx]

        self.logger.info(f"Optimal Threshold (maximizing F1): {optimal_threshold:.4f}")
        self.logger.info(f"Optimal F1-Score at this threshold: {optimal_f1_score:.4f}")
        self.logger.info(f"Precision at optimal threshold: {optimal_precision:.4f}")
        self.logger.info(f"Recall at optimal threshold: {optimal_recall:.4f}")

        # Plotting the PR curve (Optional, requires matplotlib)
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker='.')
        plt.plot(optimal_recall, optimal_precision, 'ro', markersize=8, label=f'Optimal (F1) Threshold: {optimal_threshold:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Combined Validation Folds)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.config.get('output_dir', 'outputs'), 'pr_curve_combined_validation.png'))
        plt.close()

        combined_preds_at_optimal_threshold = (np.array(combined_predicted_probs) >= optimal_threshold).astype(int)
        optimal_bal_acc = balanced_accuracy_score(combined_true_labels, combined_preds_at_optimal_threshold)
        self.logger.info(f"Balanced Accuracy at optimal threshold: {optimal_bal_acc:.4f}")
        
        # Final evaluation on test set
        self._final_test_evaluation(test_df, data_root, optimal_threshold)
        
        return fold_results
    
    def _train_fold(self, train_loader: DataLoader, val_loader: DataLoader, fold: int, start_epoch: int = 0) -> Tuple[Dict[str, float], List[int], List[float]]:
        """Train a single fold."""

        # Combine model and patch_producer parameters
        params = list(self.model.parameters())
        if self.patch_producer:
            params += list(self.patch_producer.parameters())

        # Setup optimizer and scheduler
        # optimizer = optim.Adam(
        #     params,
        #     lr=self.config.get('learning_rate', 1e-4),
        #     weight_decay=self.config.get('weight_decay', 1e-4)
        # )

        optimizer = optim.AdamW(
            params,
            lr=self.config['learning_rate'], # This will be our *target* LR after warm-up
            weight_decay=self.config['weight_decay']
        )

        # Load model and optimizer state if resuming this fold
        if start_epoch > 0:
            # Find the checkpoint for this fold and epoch
            checkpoint_pattern = os.path.join(
                self.config.get('output_dir', 'outputs'),
                f'best_model_fold_{fold}_epoch_*.pth'
            )
            found_checkpoints = glob.glob(checkpoint_pattern)
            
            # Find the checkpoint with the highest epoch number up to start_epoch - 1
            # (since start_epoch is the epoch we *start* training, so we need the model from before it)
            resume_model_path = None
            max_found_epoch = -1
            for cp_path in found_checkpoints:
                match = re.search(r'_epoch_(\d+)\.pth$', cp_path)
                if match:
                    cp_epoch = int(match.group(1)) - 1 # 0-indexed epoch
                    if cp_epoch < start_epoch and cp_epoch > max_found_epoch:
                        max_found_epoch = cp_epoch
                        resume_model_path = cp_path
            
            if resume_model_path:
                self.logger.info(f"Loading checkpoint for Fold {fold} from {resume_model_path}")
                checkpoint = torch.load(resume_model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                # Note: We are not saving/loading optimizer state currently.
                # For true resume, you'd save optimizer.state_dict() and load it here.
                # Example: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # This requires saving optimizer state in the checkpoint.
            else:
                self.logger.warning(f"No suitable checkpoint found for Fold {fold} to resume from epoch {start_epoch}. Starting fold from scratch.")
                start_epoch = 0 # Reset to 0 if no checkpoint found
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5,
            # verbose=True
        )
        
        # Setup loss function
        if self.config['soft_label']:
            if self.config['class_weight']:
                # Step 1: Compute the ratio of negatives to positives
                pos_count = sum(y == 1 for _, y, _ in train_loader.dataset)
                neg_count = sum(y == 0 for _, y, _ in train_loader.dataset)
                pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(self.device)
    
                # Step 2: Use in BCEWithLogitsLoss
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = nn.BCEWithLogitsLoss()

        else:
            if self.config['focal_loss']:
                criterion = FocalLoss(alpha=0.25, gamma=2) # Common starting point
            elif self.config['class_weight']:
                # === 1. Compute class weights ===
                targets = [label for _, label, _ in train_loader.dataset]  # Assumes __getitem__ returns (image, label, meta)
                class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=targets)
                weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
                criterion = nn.CrossEntropyLoss(weight=weight_tensor)   #  handle class imbalance by using class weights in the loss function to penalize the model more for misclassifying the minority class (cancer).
            else:
                criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_metrics = None
        best_val_score = 0
        patience_counter = 0
        max_patience = self.config.get('patience', 10)

        warmup_steps = len(train_loader) * self.config['warmup_epochs']

        # Add these new lists to store true labels and probabilities for the best model of the fold
        best_val_true_labels = []
        best_val_predicted_probs = []

        previous_best_model_path = None # Will store the path of the last saved best model for deletio

        # If resuming, find the path of the model saved at max_found_epoch
        if start_epoch > 0 and resume_model_path:
            previous_best_model_path = resume_model_path
            # Also, load the best_val_score from the CSV for this fold up to max_found_epoch
            csv_file = f"{self.config['output_dir']}/fold_{fold}_metrics.csv"
            if os.path.exists(csv_file):
                df_metrics = pd.read_csv(csv_file)
                if not df_metrics.empty:
                    # Find the row corresponding to max_found_epoch + 1 (1-indexed epoch)
                    best_epoch_row = df_metrics[df_metrics['Epoch'] == max_found_epoch + 1]
                    if not best_epoch_row.empty:
                        # Assuming default_metric is in val_metrics 
                        # Use the mapping to get the correct CSV column name
                        csv_metric_name = self.METRIC_TO_CSV_COL_MAP.get(self.config['default_metric'], self.config['default_metric'])
                        column_key = f"Val {csv_metric_name}"
                        best_val_score = best_epoch_row[column_key].iloc[0]
                        self.logger.info(f"Loaded best_val_score for Fold {fold} from CSV: {best_val_score:.4f}")
        
        # Epoch loop starts from start_epoch
        for epoch in range(start_epoch, self.config['epochs']):
            # Train
            train_loss, train_metrics = self.train_epoch(
                self.model, train_loader, optimizer, criterion, epoch, warmup_steps
            )
            
            # Validate
            val_loss, val_metrics, current_val_true, current_val_probs = self.validate_epoch(
                self.model, val_loader, criterion, epoch
            )
            
            # Check for improvement
            # Get validation score: average over selected default metrics
            if isinstance(self.config['default_metric'], list):
                val_score = np.mean([
                    val_metrics[m] for m in self.config['default_metric'] if m in val_metrics
                ])
            else:
                val_score = val_metrics[self.config['default_metric']]

            # Update scheduler
            # --- Scheduler Step (after each epoch) ---
            # Only step the plateau scheduler after the warm-up period.
            if epoch >= self.config['warmup_epochs']:
                scheduler.step(val_score)
            # --- End Scheduler Step ---

            if val_score > best_val_score:
                best_val_score = val_score
                best_val_metrics = val_metrics.copy()
                patience_counter = 0

                # --- Delete previous best model if it exists ---
                if previous_best_model_path and os.path.exists(previous_best_model_path):
                    try:
                        os.remove(previous_best_model_path)
                        self.logger.info(f"Deleted old best model: {previous_best_model_path}")
                    except OSError as e:
                        self.logger.warning(f"Error deleting old best model {previous_best_model_path}: {e}")
                
                # Save best model
                model_path = os.path.join(
                    self.config.get('output_dir', 'outputs'), 
                    f'best_model_fold_{fold}_epoch_{epoch+1}.pth' # Added _epoch_{epoch+1}
                )
                # For true resume, you should also save optimizer state:
                # torch.save({
                #     'model_state_dict': self.model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'epoch': epoch
                # }, model_path)
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"Saved new best model: {model_path}")

                # Update the path to the newly saved model
                previous_best_model_path = model_path
                
                # --- Store these for threshold tuning later ---
                best_val_true_labels = current_val_true
                best_val_predicted_probs = current_val_probs
            else:
                patience_counter += 1
                
            # Add these debug statements IMMEDIATELY before your self.logger.info block
            # print("--- DEBUGGING METRICS TYPES ---")
            # for key, value in train_metrics.items():
            #     print(f"Train Metric '{key}': Type={type(value)}, Value={value}")
            # for key, value in val_metrics.items():
            #     print(f"Val Metric '{key}': Type={type(value)}, Value={value}")
            # print("-------------------------------")
                        
            # Log progress
            self.logger.info(
                f'Fold {fold} Epoch {epoch+1}: '
                f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                f'Train Acc: {train_metrics["accuracy"]:.4f}, '
                f'Train Balanced Acc: {train_metrics["balanced_accuracy"]:.4f}, '
                f'Train pF1: {train_metrics["pF1"]:.4f}, '
                f'Train MacroF1: {train_metrics["macroF1"]:.4f}, '
                f'Train AUC: {train_metrics["auc_roc"]:.4f}, '
                f'Train Recall: {train_metrics["recall"]:.4f}, '
                f'Train Precision: {train_metrics["precision"]:.4f}, '
                f'Val Acc: {val_metrics["accuracy"]:.4f}, '
                f'Val Balanced Acc: {val_metrics["balanced_accuracy"]:.4f}, '
                f'Val pF1: {val_metrics["pF1"]:.4f}, '
                f'Val MacroF1: {val_metrics["macroF1"]:.4f}, '
                f'Val AUC: {val_metrics["auc_roc"]:.4f}, '
                f'Val Recall: {val_metrics["recall"]:.4f}, '
                f'Val Precision: {val_metrics["precision"]:.4f}'
            )

            # CSV file path for the current fold
            csv_file = f"{self.config['output_dir']}/fold_{fold}_metrics.csv"

            # If this is the first epoch being written for this fold, create the file and write the header
            # Or if resuming and this is the first epoch of the resume, ensure header is there if file is new
            # If file exists and we are resuming, we append.
            write_header = not os.path.exists(csv_file)
            if write_header:
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
        
        # Currently: return best_val_metrics
        # New: return best_val_metrics, best_val_true_labels, best_val_predicted_probs
        return best_val_metrics, best_val_true_labels, best_val_predicted_probs
    
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
    def _final_test_evaluation(self, test_df: pd.DataFrame, data_root: str, optimal_threshold: float):
        """Evaluate on final test set using ensemble of all fold models."""
        self.logger.info("\nFinal Test Set Evaluation")
        self.logger.info("-" * 30)
        
        # Save test split
        test_path = os.path.join(self.config.get('output_dir', 'outputs'), 'final_test_split.csv')
        test_df.to_csv(test_path, index=False)

        # if 'predicted_class' in test_df.columns:
        #     test_df = test_df.drop(columns=['predicted_class'])
        # test_df = test_df.drop(columns=['predicted_probability'])
        # test_df.to_csv(test_path, index=False)
        
        # Create test dataset
        # test_dataset = BreastCancerDataset(test_path, data_root, self.val_transform)
        test_dataset = BreastCancerDatasetMV(csv_path=test_path, data_root=data_root, transform=self.val_transform, target_col=self.config.get('target_col', 'cancer'), include_metadata=self.include_metadata, data_root_external=None)

        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.get('batch_size', 32),
            shuffle=False, 
            num_workers=self.config.get('num_workers', 4),
            collate_fn=self.mv_collate
        )
        
        # Load all fold models
        models = []
        k_folds = self.config.get("k_folds", 5)
        for fold in range(1, k_folds + 1):
            # Assuming only ONE such file exists per fold due to your saving/deletion logic
            search_pattern = os.path.join(
                self.config.get('output_dir', 'outputs'),
                f'best_model_fold_{fold}_epoch_*.pth'
            )
        
            # Find all files matching the pattern.
            # We expect this list to contain at most one file.
            found_models = glob.glob(search_pattern)
        
            if not found_models:
                self.logger.warning(f"No best model found for fold {fold} matching pattern '{search_pattern}'. Skipping.")
                continue # Skip to the next fold if no model is found
        
            if len(found_models) > 1:
                self.logger.warning(f"Multiple best models found for fold {fold}: {found_models}. Loading the first one found.")
                # If this happens, your deletion logic might not be working as intended.
                # For now, we'll just pick the first one.
                model_path = found_models[0]
            else:
                model_path = found_models[0]
                
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
        all_breast_ids = [] # To store image_ids for later merging

        if self.config['soft_label']:
            criterion = nn.BCEWithLogitsLoss()
        else:
            if self.config['focal_loss']:
                criterion = FocalLoss(alpha=0.25, gamma=2) # Common starting point
            else:
                criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Test [Ensemble]')
            # for batch_idx, (images, targets, metadata, image_ids) in enumerate(pbar):
            for batch_idx, (images_CC, images_MLO, targets, metadata, breast_ids) in enumerate(pbar):
                # images_CC, images_MLO, targets, metadata = images_CC.to(self.device), images_MLO.to(self.device), targets.to(self.device), metadata.to(self.device)
                targets = targets.to(self.device)

                images_CC = [[view.to(self.device) for view in sample] for sample in images_CC]
                images_MLO = [[view.to(self.device) for view in sample] for sample in images_MLO]

                # patch modify
                if self.patch_producer:
                    patch = self.patch_producer(metadata)        # (B, C, 16, W_patch)
                    patch = patch.to(self.device)

                    # apply to every CC view
                    for i in range(images_CC.shape[1]):
                        images_CC[:, i, :, :16, 208:] = patch

                    # apply to every MLO view
                    for j in range(images_MLO.shape[1]):
                        images_MLO[:, j, :, :16, 208:] = patch

                # images_CC  = [images_CC[:, i] for i in range(images_CC.size(1))]
                # images_MLO = [images_MLO[:, j] for j in range(images_MLO.size(1))]

                outputs_list = [model(images_MLO, images_CC) for model in models]
                ensemble_outputs = torch.stack(outputs_list).mean(dim=0)

                if self.config['soft_label']:
                    soft_targets = targets.float().clone()
                    soft_targets[soft_targets == 1] = self.config['soft_pos']
                    soft_targets[soft_targets == 0] = self.config['soft_neg']
                    # soft_targets = soft_targets.unsqueeze(1)
                    loss = criterion(ensemble_outputs.squeeze(1), soft_targets)
                else:
                    loss = criterion(ensemble_outputs, targets)
      

                total_loss += loss.item()

                # Get predictions
                if self.config['soft_label'] or self.config['focal_loss']:
                    probs = torch.sigmoid(ensemble_outputs)            # Correct for BCEWithLogitsLoss
                    # preds = (probs >= self.config['threshold']).long()             # or tune this threshold if needed
                    preds = (probs >= optimal_threshold).long()
                else:
                    probs = torch.softmax(ensemble_outputs, dim=1)
                    preds = torch.argmax(ensemble_outputs, dim=1)

                preds = preds.squeeze()

                # final_test_split = os.path.join(self.config.get('output_dir', 'outputs'), 'final_test_split.csv')
                # add the probs and preds column to this csv file (to the row the has image_id = test_df['image_id'], dont delete anything else in the csv file
                # Append batch results
                # all_breast_ids.extend(breast_ids.cpu().numpy()) # Assuming image_ids are accessible and can be converted to numpy
                all_breast_ids.extend(np.array(breast_ids))
                # print(preds.shape)
                # all_preds.extend(preds.cpu().numpy())
                preds_np = preds.cpu().numpy()

                # If preds_np is a scalar (0-d array), wrap it in a list
                if preds_np.ndim == 0:
                    all_preds.append(preds_np.item())  # .item() converts it to a native Python scalar
                else:
                    all_preds.extend(preds_np)
                all_targets.extend(targets.cpu().numpy())

                if self.config['soft_label'] or self.config['focal_loss']:
                    all_probs.extend(probs.detach().cpu().numpy())  # probs is already class 1
                else:
                    all_probs.extend(probs[:, 1].cpu().numpy())  # take prob for class 1
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })

        # Create a DataFrame from the collected predictions and probabilities
        results_df = pd.DataFrame({
            'breast_id': all_breast_ids,
            'predicted_probability': [p[1] if len(p) > 1 else p[0] for p in all_probs], # Adjust if not binary classification
            'predicted_class': all_preds
        })

        # Merge the results with the original test_df
        # Ensure 'image_id' is the common column for merging
        test_df_with_preds = pd.merge(test_df, results_df, on='breast_id', how='left')

        # Save the updated CSV
        test_df_with_preds.to_csv(test_path, index=False)
        self.logger.info(f"Updated final_test_split.csv with predictions and probabilities at {test_path}")

        
        test_loss = total_loss / len(test_loader)
        test_metrics = MetricsCalculator.calculate_metrics(all_targets, all_preds, all_probs)

        # Convert test_metrics to pure Python types
        test_metrics = {
            k: (v.item() if isinstance(v, np.generic) else float(v))
            for k, v in test_metrics.items()
        }
        
        self.logger.info(f"Test Loss: {test_loss:.4f}")
        self.logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        self.logger.info(f"Test pF1: {test_metrics['pF1']:.4f}")
        self.logger.info(f"Test macroF1: {test_metrics['macroF1']:.4f}")
        self.logger.info(f"Test AUC-ROC: {test_metrics.get('auc_roc', 0.0):.4f}")
        self.logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
        self.logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        # self.logger.info(f"Test Accuracy: {float(test_metrics['accuracy']):.4f}")
        # self.logger.info(f"Test Balanced Accuracy: {float(test_metrics['balanced_accuracy']):.4f}")
        # self.logger.info(f"Test pF1: {float(test_metrics['pF1']):.4f}")
        # self.logger.info(f"Test macroF1: {float(test_metrics['macroF1']):.4f}")
        # self.logger.info(f"Test AUC-ROC: {float(test_metrics.get('auc_roc', 0.0)):.4f}")
        # self.logger.info(f"Test Recall: {float(test_metrics['recall']):.4f}")
        # self.logger.info(f"Test Precision: {float(test_metrics['precision']):.4f}")

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