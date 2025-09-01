"""
Advanced Image Preprocessing for Signature Verification
Focuses on extracting distinctive signature features
"""

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.scaler = StandardScaler()
        
    def preprocess_signature(self, image_path):
        """
        Advanced preprocessing pipeline for distinctive feature extraction
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            original_shape = image.shape
            
            # Step 1: Advanced noise reduction
            denoised = self._advanced_denoising(image)
            
            # Step 2: Contrast enhancement with CLAHE
            enhanced = self._enhance_contrast(denoised)
            
            # Step 3: Signature extraction and isolation
            extracted = self._extract_signature_region(enhanced)
            
            # Step 4: Normalize and standardize
            normalized = self._normalize_signature(extracted)
            
            # Step 5: Convert to RGB and resize
            final_image = self._resize_and_format(normalized)
            
            return final_image
            
        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {e}")
            raise
    
    def _advanced_denoising(self, image):
        """Multi-stage noise reduction"""
        
        # 1. Bilateral filter for edge preservation
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 2. Non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(bilateral, None, 10, 10, 7, 21)
        
        # 3. Morphological opening to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def _enhance_contrast(self, image):
        """Advanced contrast enhancement"""
        
        # Convert to LAB color space for better contrast control
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        
        # Merge back
        enhanced_lab = cv2.merge((enhanced_l, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _extract_signature_region(self, image):
        """Extract signature region with better background removal"""
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding for better separation
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert so signature is white on black
        binary = cv2.bitwise_not(binary)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (main signature)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add intelligent padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Extract region
            signature_region = image[y:y+h, x:x+w]
            
            # Ensure minimum size
            if signature_region.shape[0] >= 50 and signature_region.shape[1] >= 50:
                return signature_region
        
        # Fallback: return center crop
        h, w = image.shape[:2]
        center_h, center_w = h // 2, w // 2
        crop_size = min(h, w) // 2
        
        start_h = max(0, center_h - crop_size)
        start_w = max(0, center_w - crop_size)
        end_h = min(h, center_h + crop_size)
        end_w = min(w, center_w + crop_size)
        
        return image[start_h:end_h, start_w:end_w]
    
    def _normalize_signature(self, signature_image):
        """Advanced normalization"""
        
        # Convert to grayscale for consistent processing
        if len(signature_image.shape) == 3:
            gray = cv2.cvtColor(signature_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = signature_image.copy()
        
        # Normalize intensity
        normalized = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur for smoothing
        smoothed = cv2.GaussianBlur(normalized, (3, 3), 0)
        
        # Threshold to get clean binary image
        _, binary = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Ensure signature is dark on light background
        if np.mean(binary) > 127:  # More white pixels than black
            binary = cv2.bitwise_not(binary)
        
        return binary
    
    def _resize_and_format(self, binary_image):
        """Resize and format for neural network input"""
        
        # Resize maintaining aspect ratio
        h, w = binary_image.shape
        target_h, target_w = self.target_size
        
        # Calculate scaling to fit within target size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(binary_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Create canvas and center the image
        canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
        
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        
        # Normalize to [0, 1] and add batch dimension
        normalized = rgb_image.astype(np.float32) / 255.0
        
        return np.expand_dims(normalized, axis=0)
    
    def batch_preprocess(self, image_paths):
        """Process multiple images efficiently"""
        processed_images = []
        failed_paths = []
        
        for img_path in image_paths:
            try:
                processed = self.preprocess_signature(img_path)
                processed_images.append(processed[0])  # Remove batch dimension
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                failed_paths.append(img_path)
                continue
        
        if processed_images:
            return np.array(processed_images), failed_paths
        else:
            return None, failed_paths
    
    def extract_handcrafted_features(self, image_path):
        """Extract additional handcrafted features for better distinction"""
        
        try:
            # Load and preprocess
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return np.zeros(20)
            
            # Get binary version
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return np.zeros(20)
            
            # Get main contour
            main_contour = max(contours, key=cv2.contourArea)
            
            # Extract geometric features
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(main_contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Convex hull
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Moments
            moments = cv2.moments(main_contour)
            if moments['m00'] > 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
            else:
                cx = cy = 0
            
            # Hu moments (rotation invariant)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # Compile features
            features = np.array([
                area / 10000,  # Normalize
                perimeter / 1000,
                aspect_ratio,
                solidity,
                cx / image.shape[1],  # Normalize by image size
                cy / image.shape[0],
                w / image.shape[1],
                h / image.shape[0],
                *hu_moments[:7],  # First 7 Hu moments
                len(contours),  # Number of connected components
                np.std(image.flatten()) / 255,  # Texture measure
                np.mean(image.flatten()) / 255,  # Average intensity
                cv2.Laplacian(image, cv2.CV_64F).var() / 10000,  # Sharpness
                (area / (w * h)) if w * h > 0 else 0  # Extent
            ])
            
            # Handle NaN values
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {image_path}: {e}")
            return np.zeros(20)