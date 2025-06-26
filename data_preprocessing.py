# import os
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pydicom
import cv2
from pathlib import Path
import logging
from typing import Optional, Tuple, Union
from scipy.ndimage import label, find_objects


class DataPreprocessor:
    def __init__(self, data_path: str, output_path: str, resize_to: Tuple[int, int] = (512, 512), 
                 crop: bool = True, apply_voilut: bool = True, stretch: bool = True, attach_patch: bool = False):
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
        self.stretch = stretch
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
        # log_file = self.output_path / 'preprocessing.log'
        log_file = '../preprocessing_all.log'
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
                # arr = dicom.pixel_array.astype(np.float64)  # Ensure float for processing, original was uint16. No need anymore
                arr = apply_voi_lut(dicom.pixel_array, dicom) # # output float64, but still integer values
                
                # Fixed variable name bug: was 'data', should be 'arr'
                if dicom.PhotometricInterpretation == "MONOCHROME1":
                    arr = arr.max() - arr  # Proper inversion for MONOCHROME1
            else:
                # Alternative implementation using dicomsdl (commented out)
                self.logger.warning("Non-VOI LUT processing not fully implemented")
                dicom = pydicom.dcmread(path)
                arr = dicom.pixel_array.astype(np.float32)
                
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

    # def crop_image(self, arr: np.ndarray, threshold_percentile: float = 5.0) -> np.ndarray:
    #     """
    #     Crop the image to remove near-uniform background.
        
    #     Args:
    #         arr: Input array
    #         threshold_percentile: Percentile threshold for background detection
            
    #     Returns:
    #         Cropped array
    #     """
    #     self.logger.debug("Cropping image...")

    #     # Use percentile-based threshold instead of minimum
    #     threshold = np.percentile(arr, threshold_percentile)
    #     mask = arr > threshold
    #     coords = np.argwhere(mask)

    #     if coords.size == 0:
    #         self.logger.warning("No valid pixels found after thresholding, returning original")
    #         return arr

    #     y0, x0 = coords.min(axis=0)         # top left
    #     y1, x1 = coords.max(axis=0)         # bottom right
        
    #     # Add small padding if possible
    #     padding = 2     # 5
    #     h, w = arr.shape
    #     y0 = max(0, y0 - padding)
    #     x0 = max(0, x0 - padding)
    #     y1 = min(h - 1, y1 + padding)
    #     x1 = min(w - 1, x1 + padding)
        
    #     return arr[y0:y1+1, x0:x1+1]

    # def crop_image(self, arr: np.ndarray,
    #             threshold_percentile: float = 1.0,
    #             horizontal_padding: int = 10,
    #             vertical_padding: int = 10) -> np.ndarray:
    #     """
    #     Crop image by finding the largest consecutive block of non-zero mean rows and columns.
    #     Removes useless background while keeping full breast area with padding.

    #     Args:
    #         arr: 2D image array
    #         horizontal_padding: Extra pixels to keep on left/right
    #         vertical_padding: Extra pixels to keep on top/bottom

    #     Returns:
    #         Cropped array
    #     """
    #     self.logger.debug("Cropping image using longest consecutive nonzero row/col blocks...")

    #     threshold = np.percentile(arr, threshold_percentile)

    #     h, w = arr.shape

    #     # ----- Horizontal (columns) -----
    #     col_means = np.mean(arr, axis=0)
    #     col_nonzero = (col_means > threshold).astype(int)

    #     longest_x = (0, 0)
    #     start = None
    #     for i, v in enumerate(col_nonzero):
    #         if v == 1:
    #             if start is None:
    #                 start = i
    #         else:
    #             if start is not None:
    #                 run = (start, i - 1)
    #                 if run[1] - run[0] > longest_x[1] - longest_x[0]:
    #                     longest_x = run
    #                 start = None
    #     if start is not None:
    #         run = (start, w - 1)
    #         if run[1] - run[0] > longest_x[1] - longest_x[0]:
    #             longest_x = run
    #     x0 = max(0, longest_x[0] - horizontal_padding)
    #     x1 = min(w - 1, longest_x[1] + horizontal_padding)

    #     # ----- Vertical (rows) -----
    #     row_means = np.mean(arr, axis=1)
    #     # row_threshold = np.percentile(row_means, 10.0)  # or 2–10 depending on aggressiveness
    #     row_nonzero = (row_means > 50).astype(int)

    #     # using standard deviation to find the breast area, not as good as mean
    #     # row_stds = np.std(arr, axis=1)
    #     # threshold = np.percentile(row_stds, 6.5)  # could also be a small constant like 1.0
    #     # row_nonflat = (row_stds > threshold).astype(int)

    #     longest_y = (0, 0)
    #     start = None
    #     for i, v in enumerate(row_nonzero):
    #         if v == 1:
    #             if start is None:
    #                 start = i
    #         else:
    #             if start is not None:
    #                 run = (start, i - 1)
    #                 if run[1] - run[0] > longest_y[1] - longest_y[0]:
    #                     longest_y = run
    #                 start = None
    #     if start is not None:
    #         run = (start, h - 1)
    #         if run[1] - run[0] > longest_y[1] - longest_y[0]:
    #             longest_y = run
    #     y0 = max(0, longest_y[0] - vertical_padding)
    #     y1 = min(h - 1, longest_y[1] + vertical_padding)

    #     return arr[y0:y1+1, x0:x1+1]
    
    # the best cropping method so far, using connected components. Use library, dont write your own, lol
    def crop_image(self, arr: np.ndarray,
                      intensity_threshold: int = 300,
                      padding: int = 50) -> np.ndarray:
        """
        Crop to the largest bright blob (likely the breast region) using connected components.
        
        Args:
            arr: 2D mammogram array (uint16 or float)
            intensity_threshold: Value above which pixels are considered breast
            padding: Pixels to add around bounding box
        
        Returns:
            Cropped image array
        """
        # Step 1: Create binary mask
        binary = arr > intensity_threshold

        # Step 2: Label connected regions
        labeled, num_features = label(binary)

        if num_features == 0:
            print("No regions found, returning original")
            return arr

        # Step 3: Find bounding boxes of all regions
        objects = find_objects(labeled)

        # Step 4: Pick the largest region by area
        max_area = 0
        best_slice = None
        for sl in objects:
            y_slice, x_slice = sl
            area = (y_slice.stop - y_slice.start) * (x_slice.stop - x_slice.start)
            if area > max_area:
                max_area = area
                best_slice = sl

        # Step 5: Apply padding
        y_slice, x_slice = best_slice
        y0 = max(0, y_slice.start - padding)
        y1 = min(arr.shape[0], y_slice.stop + padding)
        x0 = max(0, x_slice.start - padding)
        x1 = min(arr.shape[1], x_slice.stop + padding)

        return arr[y0:y1, x0:x1]
    
    # this will not strectch the image, but the breast area will be much smaller
    def resize_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Pad the array to square shape with minimal padding, then resize to target dimensions.

        Args:
            arr: Input array

        Returns:
            Resized square array
        """
        self.logger.debug(f"Resizing array from {arr.shape} to {self.resize_to}")

        if (not self.stretch):
            # Step 1: Pad to square
            h, w = arr.shape
            diff = abs(h - w)

            # Calculate top/bottom or left/right padding
            if h > w:
                pad_left = diff // 2
                pad_right = diff - pad_left
                pad_top = pad_bottom = 0
            else:
                pad_top = diff // 2
                pad_bottom = diff - pad_top
                pad_left = pad_right = 0

            # Pad with constant black (0.0)
            arr = cv2.copyMakeBorder(
                arr,
                pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=0.0
            )

            self.logger.debug(f"Padded array to square: {arr.shape}")

        # Step 2: Convert to float32 if needed
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        # Step 3: Resize to target size
        resized = cv2.resize(arr, self.resize_to, interpolation=cv2.INTER_AREA)
        return resized

    def apply_nlm_denoising(self, arr_uint16: np.ndarray, h: float = 10.0, template_window_size: int = 7, search_window_size: int = 21) -> np.ndarray:  # 4, 11, 22
        """        
        Apply Non-Local Means denoising on float32 array.

        Args:
            arr_uint16: Input array in uint16 format
            h: Filtering strength
            template_window_size: Size in pixels of the template patch
            search_window_size: Size in pixels of the window used to compute weighted average

        Returns:
            Denoised array
        """

        # comment out the following line if you want to use the default h value
        std = np.std(arr_uint16)
        h = np.clip(std / 100, 4.0, 10.0)
        
        denoised = cv2.fastNlMeansDenoising(
            src=arr_uint16,
            h=[h],
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size,
            normType=cv2.NORM_L1
            # normType = 1
        )

        print("In denoising function: ", denoised.dtype)

        return denoised

    def apply_clahe(self, arr_uint16 : np.ndarray, clip_limit : float = 6.0, tile_grid_size : Tuple[int, int] = (32,16)) -> np.ndarray:   # 4.5, (32, 32) # (8,8) results in less sharpness and contrast (more smoothness)
        """        
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on uint16 array.

        Args:
            arr_uint16: Input array in uint16 format
            clip_limit: Contrast limit for CLAHE
            tile_grid_size: Size of grid for CLAHE

        Returns:
            CLAHE processed array in uint16 format    
        """
        # print(arr_uint16.dtype, arr_uint16.shape, arr_uint16.ndim)

        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        clahe_uint16 = clahe.apply(arr_uint16)

        print("In CLAHE function: ", clahe_uint16.dtype)
        
        return clahe_uint16
    
        # arr_uint8 = (arr_uint16 / arr_uint16.max() * 255).astype(np.uint8)

        # # Apply CLAHE
        # clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        # clahe_uint8 = clahe.apply(arr_uint8)

        # # Convert back to uint16, scaling up
        # clahe_uint16 = (clahe_uint8.astype(np.float32) / 255 * arr_uint16.max()).astype(np.uint16)

        # return clahe_uint16
    
    def apply_bilateral_filter(self, arr : np.ndarray, d : int = 5, sigma_color : int = 20, sigma_space : int = 20) -> np.ndarray:  # 5, 20, 20
        """        
        Apply bilateral filter on float32 array.
        
        Args:            
            arr: Input array in float32 format
            d: Diameter of pixel neighborhood   
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space
        
        Returns:
            Filtered array in float32 format
        """
        # Ensure float32
        arr_float32 = arr.astype(np.float32)
        
        # Apply bilateral filter directly on float data
        filtered = cv2.bilateralFilter(
            arr_float32, 
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space
        )

        print("In bilateral filter function: ", filtered.dtype)

        return filtered
    
    def enhance_image(self, arr: np.ndarray, 
                 apply_nlm: bool = True, 
                 apply_clahe: bool = True, 
                 apply_bilateral: bool = True) -> np.ndarray:
        """
        Apply image enhancement techniques while preserving float precision.

        Args:
            arr: Input array
            apply_nlm: Whether to apply Non-Local Means denoising
            apply_clahe: Whether to apply CLAHE
            apply_bilateral: Whether to apply bilateral filtering

        Returns:
            Enhanced array in uint16 format
        """
        # enhanced = arr.copy()

        # Ensure array is in proper format for cv2
        if arr.dtype != np.uint16:
            # print("Oiiiii")
            arr = arr.astype(np.uint16)

        if apply_clahe:
            self.logger.debug("Applying CLAHE...")
            arr = self.apply_clahe(arr)
        
        if apply_nlm:
            self.logger.debug("Applying NL-means denoising...")
            arr = self.apply_nlm_denoising(arr)
        
        if apply_bilateral:
            self.logger.debug("Applying bilateral filter...")
            arr = self.apply_bilateral_filter(arr)
        
        return arr

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

            # Apply enhancements BEFORE normalization
            arr = self.enhance_image(arr)

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
        Arrays are saved as .npy files float32 format.
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