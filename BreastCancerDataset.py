import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import os
import albumentations as A

class BreastCancerDataset(Dataset):
    """Dataset class for breast cancer detection from mammogram scans."""
    
    def __init__(self, 
                 csv_path: str, 
                 data_root: str, 
                 transform: Optional[A.Compose] = None,
                 target_col: str = 'cancer'):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file with scan_id, patient_id, and target
            data_root: Root directory containing patient folders with .npy files
            transform: Albumentations transform pipeline
            target_col: Name of target column in CSV
        """
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.transform = transform
        self.target_col = target_col
        
        # Validate data paths
        self._validate_data_paths()
        
    def _validate_data_paths(self):
        """Validate that data files exist."""
        missing_files = []
        for idx, row in self.df.iterrows():
            file_path = self._get_file_path(row['patient_id'], row['image_id'])
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"Warning: {len(missing_files)} missing files found")
            if len(missing_files) <= 5:
                print("Missing files:", missing_files)
    
    def _get_file_path(self, patient_id: str, scan_id: str) -> str:
        """Get full file path for a scan."""
        return os.path.join(self.data_root, str(patient_id), f"{scan_id}.npy")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get a single item from the dataset.
        
        Returns:
            image: Preprocessed image tensor
            target: Target class (0 or 1)
            metadata: Dictionary with scan_id, patient_id
        """
        row = self.df.iloc[idx]
        scan_id = row['image_id']
        patient_id = row['patient_id']
        target = int(row[self.target_col])
        
        # Load image
        file_path = self._get_file_path(patient_id, scan_id)
        try:
            image = np.load(file_path)
            
            # Ensure image is in correct format
            if image.dtype != np.float32:
                image = image.astype(np.float32)

            # if isinstance(image, torch.Tensor):
            #     # print("WTFFF")
            #     image = image.numpy()
            
            # Optional: if needed by the transform
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)  # Make shape [H, W, 1]

            # print("Type before transform:", type(image))

            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']  # Albumentations returns dict

            else:
                # Convert to tensor only if no transform is applied
                image = torch.from_numpy(image).permute(2, 0, 1).float()
                
            # Ensure 3 channels for pretrained models
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return dummy data
            image = torch.zeros(3, 224, 224)
            target = 0
        
        metadata = {
            'scan_id': scan_id,
            'patient_id': patient_id,
            'file_path': file_path
        }
        
        return image, target, metadata