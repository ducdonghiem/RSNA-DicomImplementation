from BreastCancerDatasetMV import BreastCancerDatasetMV
import os
from torch.utils.data import DataLoader
import torch
import glob
import pandas as pd
import albumentations as A
import albumentations.pytorch as AP
from ModelFactory import ModelFactory
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.special import expit  # sigmoid
import numpy as np

def _get_val_transforms() -> A.Compose:
        """Get validation transforms."""
        return A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AP.ToTensorV2()
        ])

def mv_collate(batch):
        cc_list_batch = [sample[0] for sample in batch]  # list of lists
        mlo_list_batch = [sample[1] for sample in batch]
        targets = torch.tensor([sample[2] for sample in batch])
        metadata = [sample[3] for sample in batch]
        breast_ids = [sample[4] for sample in batch]

        return cc_list_batch, mlo_list_batch, targets, metadata, breast_ids

def calibrate(test_df: pd.DataFrame, data_root: str):
        """Evaluate on final test set using ensemble of all fold models."""
        
        # Save test split
        test_path = os.path.join('../outputs-correctedMV-sl059537-3000DM-best', 'temp_calibrate.csv')
        test_df.to_csv(test_path, index=False)
        
        # Create test dataset
        # test_dataset = BreastCancerDataset(test_path, data_root, self.val_transform)
        test_dataset = BreastCancerDatasetMV(csv_path=test_path, data_root=data_root, transform=_get_val_transforms(), target_col='cancer', include_metadata=False, data_root_external=None)

        test_loader = DataLoader(
            test_dataset, 
            batch_size=32,
            shuffle=False, 
            num_workers=12,
            collate_fn=mv_collate
        )
        
        # Load all fold models
        models = []
        k_folds = 5
        for fold in range(1, k_folds + 1):
            # Assuming only ONE such file exists per fold due to your saving/deletion logic
            search_pattern = os.path.join(
                '../outputs-correctedMV-sl059537-3000DM-best',
                f'best_model_fold_{fold}_epoch_*.pth'
            )
        
            # Find all files matching the pattern.
            # We expect this list to contain at most one file.
            found_models = glob.glob(search_pattern)
        
            if not found_models:
                print(f"No best model found for fold {fold} matching pattern '{search_pattern}'. Skipping.")
                continue # Skip to the next fold if no model is found
        
            if len(found_models) > 1:
                print(f"Multiple best models found for fold {fold}: {found_models}. Loading the first one found.")
                # If this happens, your deletion logic might not be working as intended.
                # For now, we'll just pick the first one.
                model_path = found_models[0]
            else:
                model_path = found_models[0]
                
            if os.path.exists(model_path):
                # fold_model = type(self.model)()  # Create new instance
                fold_model = ModelFactory.create_model(
                    'mv_model',
                    num_classes=1,
                    pretrained=False
                )
                fold_model.load_state_dict(torch.load(model_path))
                fold_model.to('cuda')
                fold_model.eval()

                models.append(fold_model)
                print(f"Loaded model from fold {fold}")
        
        # Ensemble prediction
        all_targets = []
        all_logits = []

        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Test [Ensemble]')
            # for batch_idx, (images, targets, metadata, image_ids) in enumerate(pbar):
            for batch_idx, (images_CC, images_MLO, targets, metadata, breast_ids) in enumerate(pbar):
                # images_CC, images_MLO, targets, metadata = images_CC.to(self.device), images_MLO.to(self.device), targets.to(self.device), metadata.to(self.device)
                targets = targets.to('cuda')

                images_CC = [[view.to('cuda') for view in sample] for sample in images_CC]
                images_MLO = [[view.to('cuda') for view in sample] for sample in images_MLO]

                # images_CC  = [images_CC[:, i] for i in range(images_CC.size(1))]
                # images_MLO = [images_MLO[:, j] for j in range(images_MLO.size(1))]

                outputs_list = [model(images_MLO, images_CC) for model in models]
                ensemble_outputs = torch.stack(outputs_list).mean(dim=0)

                all_logits.append(ensemble_outputs.cpu().numpy())
                
                # If preds_np is a scalar (0-d array), wrap it in a list
                all_targets.extend(targets.cpu().numpy())

        # Flatten arrays
        all_logits = np.concatenate(all_logits).reshape(-1, 1)  # shape: (N, 1)
        all_targets = np.array(all_targets)

        # Fit logistic regression for Platt scaling
        # Solver 'lbfgs' handles binary case well
        platt = LogisticRegression(solver='lbfgs')
        platt.fit(all_logits, all_targets)

        a = platt.coef_[0][0]   # slope
        b = platt.intercept_[0] # bias

        print(f"Platt scaling parameters: a = {a:.4f}, b = {b:.4f}")

        # Before and after calibration
        p_before = expit(all_logits)              # sigmoid(logits)
        p_after = expit(a * all_logits + b)       # Platt scaling

        print("Before calibration log-loss:", log_loss(all_targets, p_before))
        print("After calibration  log-loss:", log_loss(all_targets, p_after))
                    
        