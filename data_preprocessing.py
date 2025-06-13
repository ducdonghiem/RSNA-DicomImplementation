# import os
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
from pathlib import Path
import logging
from typing import Optional, Tuple, Union

class DataPreprocessor:
    def __init__(self, data_path: str, output_path: str, resize_to: Tuple[int, int] = (512, 512), 
                 crop: bool = True, apply_voilut: bool = True, attach_patch: bool = False):
        """
        Initialize DataPreprocessor for DICOM files.
        
        Args:
            data_path: Path to input DICOM files
            output_path: Path to save processed files
            resize_to: Target size as (width, height)
            crop: Whether to crop images to remove background
            apply_voilut: Whether to apply VOI LUT transformation
            attach_patch: Whether to attach patches (not implemented)
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.resize_to = resize_to
        self.crop = crop
        self.apply_voilut = apply_voilut
        self.attach_patch = attach_patch
        
        # Validate inputs
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        self.logger.info(f"DataPreprocessor initialized with data path: {self.data_path}")

    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging configuration.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"{__name__}_{id(self)}")  # Unique logger per instance
        
        # Avoid duplicate handlers if logger already configured
        if logger.handlers:
            return logger
            
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler - logs to file in output directory
        log_file = self.output_path / 'preprocessing.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # More detailed logs in file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger

    def load_data(self):
        """Placeholder for loading data logic"""
        self.logger.info("Loading data...")
        # Implement actual loading logic here

    def dcm_to_array(self, path: Union[str, Path]) -> np.ndarray:
        """
        Convert DICOM file to numpy array.
        
        Args:
            path: Path to DICOM file
            
        Returns:
            numpy array of pixel data
            
        Raises:
            Exception: If DICOM file cannot be read or processed
        """
        try:
            if self.apply_voilut:
                dicom = pydicom.dcmread(path)
                arr = dicom.pixel_array.astype(np.float64)  # Ensure float for processing
                arr = apply_voi_lut(arr, dicom)
                
                # Fixed variable name bug: was 'data', should be 'arr'
                if dicom.PhotometricInterpretation == "MONOCHROME1":
                    arr = arr.max() - arr  # Proper inversion for MONOCHROME1
            else:
                # Alternative implementation using dicomsdl (commented out)
                self.logger.warning("Non-VOI LUT processing not fully implemented")
                dicom = pydicom.dcmread(path)
                arr = dicom.pixel_array.astype(np.float64)
                
                if dicom.PhotometricInterpretation == "MONOCHROME1":
                    arr = arr.max() - arr

            return arr
            
        except Exception as e:
            self.logger.error(f"Error processing DICOM file {path}: {str(e)}")
            raise

    def min_max_normalize(self, arr: np.ndarray) -> np.ndarray:
        """
        Apply min-max normalization to array.
        
        Args:
            arr: Input array
            
        Returns:
            Normalized array with values between 0 and 1
        """
        arr_min, arr_max = arr.min(), arr.max()
        
        # Handle edge case where all values are the same
        if arr_max == arr_min:
            self.logger.warning("Array has uniform values, returning zeros")
            return np.zeros_like(arr)
            
        return (arr - arr_min) / (arr_max - arr_min)

    def crop_image(self, arr: np.ndarray, threshold_percentile: float = 1.0) -> np.ndarray:
        """
        Crop the image to remove near-uniform background.
        
        Args:
            arr: Input array
            threshold_percentile: Percentile threshold for background detection
            
        Returns:
            Cropped array
        """
        self.logger.debug("Cropping image...")

        # Use percentile-based threshold instead of minimum
        threshold = np.percentile(arr, threshold_percentile)
        mask = arr > threshold
        coords = np.argwhere(mask)

        if coords.size == 0:
            self.logger.warning("No valid pixels found after thresholding, returning original")
            return arr

        y0, x0 = coords.min(axis=0)         # top left
        y1, x1 = coords.max(axis=0)         # bottom right
        
        # Add small padding if possible
        padding = 5
        h, w = arr.shape
        y0 = max(0, y0 - padding)
        x0 = max(0, x0 - padding)
        y1 = min(h - 1, y1 + padding)
        x1 = min(w - 1, x1 + padding)
        
        return arr[y0:y1+1, x0:x1+1]

    def resize_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Resize array to target dimensions.
        
        Args:
            arr: Input array
            
        Returns:
            Resized array
        """
        self.logger.debug(f"Resizing array from {arr.shape} to {self.resize_to}")
        
        # Ensure array is in proper format for cv2
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
            
        resized = cv2.resize(arr, self.resize_to, interpolation=cv2.INTER_AREA)
        return resized

    def attach_patch_method(self):
        """Placeholder for attaching patch logic"""
        self.logger.info("Attaching patch to images...")
        # Implement actual patch attachment logic here

    def process_single_dcm(self, file_path: Path) -> Optional[np.ndarray]:
        """
        Process a single DICOM file.
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            Processed array or None if processing failed
        """
        try:
            self.logger.info(f"Processing {file_path}...")
            arr = self.dcm_to_array(file_path)
            
            if self.crop:
                arr = self.crop_image(arr)
            
            arr = self.resize_array(arr)
            arr = self.min_max_normalize(arr)
            
            return arr
            
        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {str(e)}")
            return None

    def process_dcm(self):
        """
        Main method to process all DICOM files in the data path.
        Fixed method name (was processing 'PNG' in log message but processing DCM files).
        """
        self.logger.info("Starting DICOM preprocessing...")
        
        processed_count = 0
        failed_count = 0

        for file_path in self.data_path.rglob('*.dcm'):
            arr = self.process_single_dcm(file_path)
            
            if arr is not None:
                # Create output directory structure
                relative_path = file_path.relative_to(self.data_path)
                output_subfolder = self.output_path / relative_path.parent
                output_subfolder.mkdir(parents=True, exist_ok=True)
                
                # Save processed array
                output_file = output_subfolder / f"{file_path.stem}.npy"
                np.save(output_file, arr)
                
                self.logger.info(f"Saved preprocessed data to {output_file}")
                processed_count += 1
            else:
                failed_count += 1

        self.logger.info(f"Processing complete. Processed: {processed_count}, Failed: {failed_count}")

    def get_statistics(self) -> dict:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        dcm_files = list(self.data_path.rglob('*.dcm'))
        return {
            'total_dcm_files': len(dcm_files),
            'data_path': str(self.data_path),
            'output_path': str(self.output_path),
            'resize_to': self.resize_to,
            'crop_enabled': self.crop,
            'voilut_enabled': self.apply_voilut
        }