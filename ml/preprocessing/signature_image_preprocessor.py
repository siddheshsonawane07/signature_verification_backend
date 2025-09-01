"""
Advanced Image Preprocessing for Signature Verification
Handles noise reduction, enhancement, and signature extraction
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from skimage import morphology, measure
from scipy import ndimage
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignatureImageProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.preprocessing_stats = {}
    
    def preprocess_signature(self, image_path, save_debug=False):
        """
        Complete preprocessing pipeline for signature images
        
        Args:
            image_path: Path to signature image
            save_debug: Save intermediate steps for debugging
            
        Returns:
            Preprocessed image ready for CNN input
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            logger.info(f"Processing image: {image_path}")
            original_shape = image.shape
            
            # Step 1: Noise reduction
            denoised = self._apply_noise_reduction(image)
            
            # Step 2: Convert to grayscale
            gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
            
            # Step 3: Contrast enhancement
            enhanced = self._enhance_contrast(gray)
            
            # Step 4: Adaptive thresholding
            binary = self._adaptive_thresholding(enhanced)
            
            # Step 5: Morphological operations
            cleaned = self._morphological_cleaning(binary)
            
            # Step 6: Signature extraction and cropping
            extracted = self._extract_signature_region(cleaned)
            
            # Step 7: Resize and normalize
            final_image = self._resize_and_normalize(extracted)
            
            # Save debug images if requested
            if save_debug:
                self._save_debug_images(image_path, {
                    'original': image,
                    'denoised': denoised,
                    'grayscale': gray,
                    'enhanced': enhanced,
                    'binary': binary,
                    'cleaned': cleaned,
                    'extracted': extracted,
                    'final': (final_image * 255).astype(np.uint8)
                })
            
            # Update statistics
            self.preprocessing_stats[image_path] = {
                'original_shape': original_shape,
                'final_shape': final_image.shape,
                'processing_successful': True
            }
            
            return final_image
            
        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {str(e)}")
            self.preprocessing_stats[image_path] = {
                'processing_successful': False,
                'error': str(e)
            }
            raise
    
    def _apply_noise_reduction(self, image):
        """
        Advanced noise reduction using multiple filters
        """
        # Bilateral filter for edge-preserving smoothing
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Gaussian blur for additional smoothing
        gaussian = cv2.GaussianBlur(bilateral, (3, 3), 0)
        
        # Non-local means denoising for better quality
        denoised = cv2.fastNlMeansDenoisingColored(gaussian, None, 10, 10, 7, 21)
        
        return denoised
    
    def _enhance_contrast(self, gray_image):
        """
        Enhance contrast using CLAHE and histogram equalization
        """
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_image)
        
        # Additional histogram equalization for better contrast
        equalized = cv2.equalizeHist(enhanced)
        
        # Combine both enhancements
        final_enhanced = cv2.addWeighted(enhanced, 0.7, equalized, 0.3, 0)
        
        return final_enhanced
    
    def _adaptive_thresholding(self, enhanced_image):
        """
        Apply adaptive thresholding with multiple methods
        """
        # Method 1: Gaussian adaptive threshold
        thresh1 = cv2.adaptiveThreshold(
            enhanced_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Method 2: Mean adaptive threshold
        thresh2 = cv2.adaptiveThreshold(
            enhanced_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Method 3: Otsu's thresholding
        _, thresh3 = cv2.threshold(
            enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Combine thresholds using weighted average
        combined = cv2.addWeighted(thresh1, 0.5, thresh2, 0.3, 0)
        combined = cv2.addWeighted(combined, 0.8, thresh3, 0.2, 0)
        
        # Ensure binary output
        _, final_binary = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)
        
        return final_binary
    
    def _morphological_cleaning(self, binary_image):
        """
        Clean binary image using morphological operations
        """
        # Define kernels for different operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Remove noise with opening
        opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_small)
        
        # Fill gaps with closing
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium)
        
        # Remove small artifacts
        cleaned = self._remove_small_artifacts(closed, min_area=50)
        
        return cleaned
    
    def _remove_small_artifacts(self, binary_image, min_area=50):
        """
        Remove small connected components (noise)
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )
        
        # Create output image
        cleaned = np.zeros_like(binary_image)
        
        # Keep only large components
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == i] = 255
        
        return cleaned
    
    def _extract_signature_region(self, binary_image):
        """
        Extract signature region using contour detection and bounding box
        """
        # Find contours
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            logger.warning("No contours found, returning original image")
            return binary_image
        
        # Find the largest contour (main signature)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle with padding
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add intelligent padding based on signature size
        padding_x = max(20, int(w * 0.1))
        padding_y = max(20, int(h * 0.1))
        
        # Apply padding with bounds checking
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(binary_image.shape[1] - x, w + 2 * padding_x)
        h = min(binary_image.shape[0] - y, h + 2 * padding_y)
        
        # Extract signature region
        signature_region = binary_image[y:y+h, x:x+w]
        
        # Ensure minimum size
        if signature_region.shape[0] < 50 or signature_region.shape[1] < 50:
            logger.warning("Extracted region too small, using original image")
            return binary_image
        
        return signature_region
    
    def _resize_and_normalize(self, signature_image):
        """
        Resize to target size and normalize for CNN input
        """
        # Resize to target size maintaining aspect ratio
        h, w = signature_image.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(signature_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Create canvas with target size
        canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
        
        # Center the resized image on canvas
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        
        # Normalize to [0, 1] range
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def _save_debug_images(self, original_path, image_dict):
        """
        Save intermediate processing steps for debugging
        """
        import os
        debug_dir = "../data/debug_images/"
        os.makedirs(debug_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        
        for step_name, image in image_dict.items():
            debug_path = f"{debug_dir}/{base_name}_{step_name}.png"
            
            if len(image.shape) == 4:  # Remove batch dimension if present
                image = image[0]
            
            if image.dtype == np.float32:  # Convert float to uint8
                image = (image * 255).astype(np.uint8)
            
            cv2.imwrite(debug_path, image)
        
        logger.info(f"Debug images saved to {debug_dir}")
    
    def extract_geometric_features(self, binary_image):
        """
        Extract handcrafted geometric features from signature
        """
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return np.zeros(25)  # Return zero vector if no contours
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Basic geometric properties
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Bounding rectangle features
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        bounding_area = w * h
        extent = area / bounding_area if bounding_area > 0 else 0
        
        # Moments and centroid
        moments = cv2.moments(largest_contour)
        if moments['m00'] > 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
        else:
            cx = cy = 0
        
        # Hu moments (rotation, scale, translation invariant)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Convex hull properties
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Convexity defects
        hull_indices = cv2.convexHull(largest_contour, returnPoints=False)
        if len(hull_indices) > 3:
            defects = cv2.convexityDefects(largest_contour, hull_indices)
            defect_count = len(defects) if defects is not None else 0
        else:
            defect_count = 0
        
        # Ellipse fitting
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            ellipse_area = np.pi * ellipse[1][0] * ellipse[1][1] / 4
            ellipse_ratio = area / ellipse_area if ellipse_area > 0 else 0
        else:
            ellipse_ratio = 0
        
        # Compile all geometric features
        geometric_features = np.array([
            area,                    # 0: Area
            perimeter,              # 1: Perimeter
            aspect_ratio,           # 2: Aspect ratio
            extent,                 # 3: Extent
            solidity,               # 4: Solidity
            cx, cy,                 # 5-6: Centroid
            w, h,                   # 7-8: Width, height
            bounding_area,          # 9: Bounding area
            hull_area,              # 10: Convex hull area
            defect_count,           # 11: Convexity defects
            ellipse_ratio,          # 12: Ellipse fitting ratio
            *hu_moments[:7],        # 13-19: Hu moments
            perimeter**2 / area if area > 0 else 0,  # 20: Circularity
            4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0,  # 21: Roundness
            area / (np.pi * ((w/2)**2)) if w > 0 else 0,  # 22: Form factor
            w / perimeter if perimeter > 0 else 0,  # 23: Width-perimeter ratio
            h / perimeter if perimeter > 0 else 0,  # 24: Height-perimeter ratio
        ])
        
        # Handle any NaN or infinite values
        geometric_features = np.nan_to_num(geometric_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return geometric_features
    
    def extract_stroke_features(self, binary_image):
        """
        Extract stroke-based features for signature analysis
        """
        # Skeletonization to analyze stroke structure
        skeleton = morphology.skeletonize(binary_image // 255).astype(np.uint8) * 255
        
        # Find stroke endpoints and junctions
        endpoints = self._find_stroke_endpoints(skeleton)
        junctions = self._find_stroke_junctions(skeleton)
        
        # Estimate stroke widths
        stroke_widths = self._estimate_stroke_widths(binary_image, skeleton)
        
        # Calculate stroke-based features
        stroke_features = np.array([
            len(endpoints),                    # Number of stroke endpoints
            len(junctions),                   # Number of stroke junctions
            np.mean(stroke_widths) if stroke_widths else 0,    # Average stroke width
            np.std(stroke_widths) if stroke_widths else 0,     # Stroke width variation
            np.max(stroke_widths) if stroke_widths else 0,     # Maximum stroke width
            np.min(stroke_widths) if stroke_widths else 0,     # Minimum stroke width
            len(stroke_widths),               # Total stroke segments
            np.sum(skeleton > 0),             # Total skeleton pixels
        ])
        
        return stroke_features
    
    def _find_stroke_endpoints(self, skeleton):
        """
        Find endpoints in skeleton image (pixels with only one neighbor)
        """
        # Kernel to count neighbors
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        # Count neighbors for each pixel
        neighbor_count = cv2.filter2D(skeleton, -1, kernel)
        
        # Endpoints have exactly one neighbor
        endpoints = np.where((skeleton > 0) & (neighbor_count == 255))
        
        return list(zip(endpoints[1], endpoints[0]))  # Return as (x, y) coordinates
    
    def _find_stroke_junctions(self, skeleton):
        """
        Find junctions in skeleton image (pixels with 3+ neighbors)
        """
        # Kernel to count neighbors
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        # Count neighbors for each pixel
        neighbor_count = cv2.filter2D(skeleton, -1, kernel)
        
        # Junctions have 3 or more neighbors
        junctions = np.where((skeleton > 0) & (neighbor_count >= 3 * 255))
        
        return list(zip(junctions[1], junctions[0]))  # Return as (x, y) coordinates
    
    def _estimate_stroke_widths(self, binary_image, skeleton):
        """
        Estimate stroke widths using distance transform
        """
        # Distance transform gives distance to nearest background pixel
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
        
        # Get stroke widths at skeleton points
        skeleton_points = np.where(skeleton > 0)
        stroke_widths = dist_transform[skeleton_points] * 2  # Multiply by 2 for full width
        
        # Filter out very small widths (noise)
        stroke_widths = stroke_widths[stroke_widths > 1.0]
        
        return stroke_widths.tolist()
    
    def batch_preprocess(self, image_paths, save_debug=False):
        """
        Process multiple images in batch with progress tracking
        """
        from tqdm import tqdm
        
        processed_images = []
        failed_images = []
        
        for image_path in tqdm(image_paths, desc="Processing signatures"):
            try:
                processed_img = self.preprocess_signature(image_path, save_debug)
                processed_images.append(processed_img)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                failed_images.append(image_path)
        
        success_rate = len(processed_images) / len(image_paths) * 100
        logger.info(f"Batch processing complete. Success rate: {success_rate:.2f}%")
        
        if failed_images:
            logger.warning(f"Failed to process {len(failed_images)} images: {failed_images}")
        
        return np.vstack(processed_images) if processed_images else None, failed_images
    
    def visualize_preprocessing_steps(self, image_path):
        """
        Visualize preprocessing steps for analysis and debugging
        """
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Apply each preprocessing step
        denoised = self._apply_noise_reduction(image)
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        enhanced = self._enhance_contrast(gray)
        binary = self._adaptive_thresholding(enhanced)
        cleaned = self._morphological_cleaning(binary)
        extracted = self._extract_signature_region(cleaned)
        final = self._resize_and_normalize(extracted)
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        images = [
            (cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "Original"),
            (cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB), "Denoised"),
            (gray, "Grayscale"),
            (enhanced, "Enhanced"),
            (binary, "Binary"),
            (cleaned, "Cleaned"),
            (extracted, "Extracted"),
            ((final[0] * 255).astype(np.uint8), "Final")
        ]
        
        for i, (img, title) in enumerate(images):
            axes[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"../data/debug_images/preprocessing_steps_{os.path.basename(image_path)}.png", 
                   dpi=150, bbox_inches='tight')
        plt.show()
    
    def validate_image_quality(self, image_path):
        """
        Validate if image is suitable for signature verification
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False, "Cannot load image"
            
            # Check image dimensions
            h, w = image.shape[:2]
            if h < 100 or w < 100:
                return False, "Image too small (minimum 100x100)"
            
            # Check if image is mostly blank
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            non_white_pixels = np.sum(gray < 240)
            total_pixels = h * w
            
            if non_white_pixels / total_pixels < 0.05:
                return False, "Image appears to be mostly blank"
            
            # Check for reasonable contrast
            contrast = gray.std()
            if contrast < 10:
                return False, "Image has insufficient contrast"
            
            return True, "Image quality acceptable"
            
        except Exception as e:
            return False, f"Error validating image: {str(e)}"

# Utility functions for data augmentation
class SignatureAugmentation:
    """
    Data augmentation specifically designed for signature images
    """
    
    @staticmethod
    def augment_signature(image, num_augmentations=5):
        """
        Generate augmented versions of signature preserving key characteristics
        """
        augmented_images = []
        
        for i in range(num_augmentations):
            augmented = image.copy()
            
            # 1. Small rotation (-5 to 5 degrees)
            angle = np.random.uniform(-5, 5)
            center = (image.shape[1]//2, image.shape[0]//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(augmented, rotation_matrix, 
                                     (image.shape[1], image.shape[0]), 
                                     borderValue=255)
            
            # 2. Small translation (-3 to 3 pixels)
            tx = np.random.randint(-3, 4)
            ty = np.random.randint(-3, 4)
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            augmented = cv2.warpAffine(augmented, translation_matrix, 
                                     (image.shape[1], image.shape[0]),
                                     borderValue=255)
            
            # 3. Slight scaling (0.98 to 1.02)
            scale = np.random.uniform(0.98, 1.02)
            scaled = cv2.resize(augmented, None, fx=scale, fy=scale)
            
            # Crop or pad to original size
            if scaled.shape[0] > image.shape[0] or scaled.shape[1] > image.shape[1]:
                # Crop center
                start_x = (scaled.shape[1] - image.shape[1]) // 2
                start_y = (scaled.shape[0] - image.shape[0]) // 2
                scaled = scaled[start_y:start_y+image.shape[0], 
                              start_x:start_x+image.shape[1]]
            else:
                # Pad to original size
                pad_y = (image.shape[0] - scaled.shape[0]) // 2
                pad_x = (image.shape[1] - scaled.shape[1]) // 2
                scaled = cv2.copyMakeBorder(
                    scaled, 
                    pad_y, image.shape[0]-scaled.shape[0]-pad_y,
                    pad_x, image.shape[1]-scaled.shape[1]-pad_x,
                    cv2.BORDER_CONSTANT, value=255
                )
            
            # 4. Add slight noise (very minimal)
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 2, scaled.shape).astype(np.uint8)
                augmented = cv2.add(scaled, noise)
                augmented = np.clip(augmented, 0, 255)
            else:
                augmented = scaled
            
            augmented_images.append(augmented)
        
        return augmented_images
    
    @staticmethod
    def create_forgery_samples(genuine_signatures, difficulty_level='medium'):
        """
        Create simulated forgery samples for training
        """
        forgeries = []
        
        for signature in genuine_signatures:
            if difficulty_level == 'easy':
                # Simple transformations
                forged = SignatureAugmentation._easy_forgery(signature)
            elif difficulty_level == 'medium':
                # Moderate transformations
                forged = SignatureAugmentation._medium_forgery(signature)
            else:  # hard
                # Complex transformations
                forged = SignatureAugmentation._hard_forgery(signature)
            
            forgeries.append(forged)
        
        return forgeries
    
    @staticmethod
    def _easy_forgery(signature):
        """Create easy-to-detect forgery"""
        # Simple scaling and rotation
        angle = np.random.uniform(-15, 15)
        scale = np.random.uniform(0.8, 1.2)
        
        center = (signature.shape[1]//2, signature.shape[0]//2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        forged = cv2.warpAffine(signature, matrix, 
                              (signature.shape[1], signature.shape[0]),
                              borderValue=255)
        return forged
    
    @staticmethod
    def _medium_forgery(signature):
        """Create medium difficulty forgery"""
        forged = signature.copy()
        
        # Apply perspective transformation
        h, w = signature.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = pts1 + np.random.uniform(-10, 10, pts1.shape)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        forged = cv2.warpPerspective(forged, matrix, (w, h), borderValue=255)
        
        # Add stroke modifications
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        if np.random.random() > 0.5:
            forged = cv2.morphologyEx(forged, cv2.MORPH_DILATE, kernel)
        else:
            forged = cv2.morphologyEx(forged, cv2.MORPH_ERODE, kernel)
        
        return forged
    
    @staticmethod
    def _hard_forgery(signature):
        """Create sophisticated forgery"""
        # This would involve more complex transformations
        # For now, use medium forgery with additional noise
        forged = SignatureAugmentation._medium_forgery(signature)
        
        # Add structured noise
        noise = np.random.normal(0, 5, forged.shape).astype(np.uint8)
        forged = cv2.add(forged, noise)
        forged = np.clip(forged, 0, 255)
        
        return forged

# Example usage and testing
if __name__ == "__main__":
    # Initialize processor
    processor = SignatureImageProcessor()
    
    # Test with sample image
    image_path = "./data/users/test_user/train/signature_1.jpg"
    
    # Validate image quality
    is_valid, message = processor.validate_image_quality(image_path)
    print(f"Image validation: {is_valid} - {message}")
    
    if is_valid:
        # Process image
        processed_img = processor.preprocess_signature(image_path, save_debug=True)
        print(f"Processed image shape: {processed_img.shape}")
        
        # Extract features
        binary_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        geometric_features = processor.extract_geometric_features(binary_img)
        stroke_features = processor.extract_stroke_features(binary_img)
        
        print(f"Geometric features shape: {geometric_features.shape}")
        print(f"Stroke features shape: {stroke_features.shape}")
        
        # Visualize preprocessing steps
        processor.visualize_preprocessing_steps(image_path)