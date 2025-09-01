"""
Core Signature Verification Engine
Handles user profile creation, signature verification, and adaptive thresholding
"""

import os
import pickle
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import DBSCAN
import logging
import cv2
from ml.models.siamese_network import SiameseNetwork
from ml.preprocessing.signature_image_preprocessor import SignatureImageProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignatureVerificationEngine:
    def __init__(self, model_path="../data/models/", profiles_path="../data/profiles/"):
        self.model_path = model_path
        self.profiles_path = profiles_path
        self.image_processor = SignatureImageProcessor()
        self.siamese_network = SiameseNetwork()
        self.feature_extractor = None
        
        # Ensure directories exist
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(profiles_path, exist_ok=True)
        
        # Load or initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize or load the signature verification model
        """
        try:
            # Try to load existing model
            if os.path.exists(f"{self.model_path}/feature_extractor.h5"):
                self.feature_extractor = tf.keras.models.load_model(
                    f"{self.model_path}/feature_extractor.h5"
                )
                logger.info("Loaded existing feature extractor model")
            else:
                # Build new model
                logger.info("Building new Siamese network...")
                self.siamese_network.build_siamese_model()
                self.siamese_network.compile_model()
                self.feature_extractor = self.siamese_network.feature_extractor
                logger.info("New Siamese network created")
                
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            # Fallback: create new model
            self.siamese_network.build_siamese_model()
            self.siamese_network.compile_model()
            self.feature_extractor = self.siamese_network.feature_extractor
    
    def create_user_profile(self, username: str, signature_paths: List[str]) -> Dict:
        """
        Create signature profile for user enrollment
        
        Args:
            username: Unique username
            signature_paths: List of paths to genuine signature images
            
        Returns:
            Dictionary containing user profile
        """
        logger.info(f"Creating profile for user: {username}")
        
        if len(signature_paths) < 5:
            raise ValueError("Minimum 5 signatures required for enrollment")
        
        # Validate all images first
        valid_paths = []
        for path in signature_paths:
            is_valid, message = self.image_processor.validate_image_quality(path)
            if is_valid:
                valid_paths.append(path)
            else:
                logger.warning(f"Skipping invalid image {path}: {message}")
        
        if len(valid_paths) < 5:
            raise ValueError(f"Only {len(valid_paths)} valid signatures found. Minimum 5 required.")
        
        # Extract features from all signatures
        cnn_features = []
        geometric_features = []
        stroke_features = []
        
        for sig_path in valid_paths:
            try:
                # CNN features
                processed_img = self.image_processor.preprocess_signature(sig_path)
                cnn_feature = self.feature_extractor.predict(processed_img, verbose=0)
                cnn_features.append(cnn_feature.flatten())
                
                # Handcrafted features
                binary_img = self._load_as_binary(sig_path)
                geo_feature = self.image_processor.extract_geometric_features(binary_img)
                stroke_feature = self.image_processor.extract_stroke_features(binary_img)
                
                geometric_features.append(geo_feature)
                stroke_features.append(stroke_feature)
                
            except Exception as e:
                logger.error(f"Error processing {sig_path}: {e}")
                continue
        
        if len(cnn_features) < 5:
            raise ValueError("Failed to extract features from minimum required signatures")
        
        # Convert to numpy arrays
        cnn_features = np.array(cnn_features)
        geometric_features = np.array(geometric_features)
        stroke_features = np.array(stroke_features)
        
        # Combine all features
        combined_features = np.concatenate([
            cnn_features,
            geometric_features,
            stroke_features
        ], axis=1)
        
        # Compute profile statistics
        profile = {
            'username': username,
            'enrollment_date': datetime.now().isoformat(),
            'signature_count': len(cnn_features),
            'valid_signature_paths': valid_paths,
            
            # Feature statistics
            'cnn_features': {
                'mean': np.mean(cnn_features, axis=0),
                'std': np.std(cnn_features, axis=0),
                'matrix': cnn_features
            },
            'geometric_features': {
                'mean': np.mean(geometric_features, axis=0),
                'std': np.std(geometric_features, axis=0),
                'matrix': geometric_features
            },
            'stroke_features': {
                'mean': np.mean(stroke_features, axis=0),
                'std': np.std(stroke_features, axis=0),
                'matrix': stroke_features
            },
            'combined_features': {
                'mean': np.mean(combined_features, axis=0),
                'std': np.std(combined_features, axis=0),
                'matrix': combined_features
            },
            
            # Adaptive thresholds
            'thresholds': self._compute_adaptive_thresholds(combined_features, cnn_features),
            
            # Model metadata
            'model_version': '1.0',
            'feature_extractor_hash': self._get_model_hash(),
            
            # Usage statistics
            'verification_count': 0,
            'last_verification': None,
            'false_rejections': 0,
            'false_acceptances': 0
        }
        
        # Save profile
        self._save_user_profile(username, profile)
        
        logger.info(f"Profile created for {username} with {len(cnn_features)} signatures")
        return profile
    
    def verify_signature(self, username: str, test_signature_path: str) -> Dict:
        """
        Verify signature against user's enrolled profile
        
        Args:
            username: Username to verify against
            test_signature_path: Path to signature image to verify
            
        Returns:
            Verification result dictionary
        """
        logger.info(f"Verifying signature for user: {username}")
        
        # Load user profile
        profile = self._load_user_profile(username)
        if profile is None:
            raise ValueError(f"No profile found for user: {username}")
        
        # Validate test image
        is_valid, message = self.image_processor.validate_image_quality(test_signature_path)
        if not is_valid:
            raise ValueError(f"Invalid test signature: {message}")
        
        try:
            # Extract features from test signature
            processed_img = self.image_processor.preprocess_signature(test_signature_path)
            test_cnn_features = self.feature_extractor.predict(processed_img, verbose=0).flatten()
            
            binary_img = self._load_as_binary(test_signature_path)
            test_geo_features = self.image_processor.extract_geometric_features(binary_img)
            test_stroke_features = self.image_processor.extract_stroke_features(binary_img)
            
            # Combine features
            test_combined_features = np.concatenate([
                test_cnn_features,
                test_geo_features,
                test_stroke_features
            ])
            
            # Compute similarities with enrolled signatures
            enrolled_features = profile['combined_features']['matrix']
            similarities = []
            
            for enrolled_feature in enrolled_features:
                # Multiple similarity metrics
                cosine_sim = cosine_similarity(
                    test_combined_features.reshape(1, -1),
                    enrolled_feature.reshape(1, -1)
                )[0][0]
                
                euclidean_dist = euclidean_distances(
                    test_combined_features.reshape(1, -1),
                    enrolled_feature.reshape(1, -1)
                )[0][0]
                
                # Normalized euclidean similarity
                euclidean_sim = 1 / (1 + euclidean_dist)
                
                # Weighted combination
                combined_sim = 0.7 * cosine_sim + 0.3 * euclidean_sim
                similarities.append(combined_sim)
            
            similarities = np.array(similarities)
            
            # Compute verification metrics
            max_similarity = np.max(similarities)
            avg_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            
            # Apply adaptive threshold
            threshold = profile['thresholds']['combined_threshold']
            is_genuine = max_similarity > threshold
            
            # Confidence calculation (0-100%)
            confidence = min(max_similarity * 100, 100.0)
            
            # Additional verification using CNN-only features
            cnn_similarities = []
            enrolled_cnn_features = profile['cnn_features']['matrix']
            
            for enrolled_cnn in enrolled_cnn_features:
                cnn_sim = cosine_similarity(
                    test_cnn_features.reshape(1, -1),
                    enrolled_cnn.reshape(1, -1)
                )[0][0]
                cnn_similarities.append(cnn_sim)
            
            cnn_max_similarity = np.max(cnn_similarities)
            cnn_threshold = profile['thresholds']['cnn_threshold']
            cnn_verification = cnn_max_similarity > cnn_threshold
            
            # Final decision (both methods must agree for high confidence)
            final_verification = is_genuine and cnn_verification
            
            # Adjust confidence based on agreement
            if is_genuine != cnn_verification:
                confidence *= 0.7  # Reduce confidence if methods disagree
            
            result = {
                'verified': bool(final_verification),
                'confidence': float(confidence),
                'max_similarity': float(max_similarity),
                'average_similarity': float(avg_similarity),
                'std_similarity': float(std_similarity),
                'threshold': float(threshold),
                'cnn_similarity': float(cnn_max_similarity),
                'cnn_threshold': float(cnn_threshold),
                'cnn_verified': bool(cnn_verification),
                'methods_agree': bool(is_genuine == cnn_verification),
                'timestamp': datetime.now().isoformat(),
                'signature_count_used': len(similarities)
            }
            
            # Update profile statistics
            self._update_profile_statistics(username, result)
            
            logger.info(f"Verification complete for {username}: {result['verified']}")
            return result
            
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            raise
    
    def _compute_adaptive_thresholds(self, combined_features, cnn_features):
        """
        Compute adaptive thresholds using multiple methods
        """
        thresholds = {}
        
        # 1. Statistical threshold for combined features
        combined_similarities = self._compute_intra_class_similarities(combined_features)
        combined_threshold = np.mean(combined_similarities) - 2 * np.std(combined_similarities)
        combined_threshold = np.clip(combined_threshold, 0.6, 0.9)
        
        # 2. Statistical threshold for CNN features only
        cnn_similarities = self._compute_intra_class_similarities(cnn_features)
        cnn_threshold = np.mean(cnn_similarities) - 2 * np.std(cnn_similarities)
        cnn_threshold = np.clip(cnn_threshold, 0.7, 0.95)
        
        # 3. Clustering-based threshold
        clustering_threshold = self._compute_clustering_threshold(combined_features)
        
        # 4. Conservative threshold (minimum of all methods)
        conservative_threshold = min(combined_threshold, cnn_threshold, clustering_threshold)
        
        thresholds = {
            'combined_threshold': float(combined_threshold),
            'cnn_threshold': float(cnn_threshold),
            'clustering_threshold': float(clustering_threshold),
            'conservative_threshold': float(conservative_threshold),
            'method_used': 'statistical_with_cnn_validation'
        }
        
        return thresholds
    
    def _compute_intra_class_similarities(self, features):
        """
        Compute pairwise similarities within the same class (genuine signatures)
        """
        n_samples = len(features)
        similarities = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                sim = cosine_similarity(
                    features[i].reshape(1, -1),
                    features[j].reshape(1, -1)
                )[0][0]
                similarities.append(sim)
        
        return np.array(similarities)
    
    def _compute_clustering_threshold(self, features):
        """
        Compute threshold using clustering analysis
        """
        try:
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=0.2, min_samples=2).fit(features)
            unique_labels = set(clustering.labels_)
            
            if len(unique_labels) <= 1 or -1 in unique_labels:
                # Fall back to statistical method if clustering fails
                similarities = self._compute_intra_class_similarities(features)
                return np.mean(similarities) - 1.5 * np.std(similarities)
            
            # Compute cluster centers
            cluster_centers = []
            for label in unique_labels:
                cluster_points = features[clustering.labels_ == label]
                center = np.mean(cluster_points, axis=0)
                cluster_centers.append(center)
            
            # Threshold based on minimum inter-cluster distance
            if len(cluster_centers) >= 2:
                min_distance = float('inf')
                for i in range(len(cluster_centers)):
                    for j in range(i + 1, len(cluster_centers)):
                        dist = cosine_similarity(
                            cluster_centers[i].reshape(1, -1),
                            cluster_centers[j].reshape(1, -1)
                        )[0][0]
                        min_distance = min(min_distance, dist)
                
                threshold = min_distance * 0.85  # 85% of minimum cluster distance
            else:
                # Single cluster, use intra-cluster similarity
                similarities = self._compute_intra_class_similarities(features)
                threshold = np.mean(similarities) - 1.5 * np.std(similarities)
            
            return np.clip(threshold, 0.5, 0.9)
            
        except Exception as e:
            logger.warning(f"Clustering threshold computation failed: {e}")
            # Fall back to statistical method
            similarities = self._compute_intra_class_similarities(features)
            return np.mean(similarities) - 2 * np.std(similarities)
    
    def _load_as_binary(self, image_path):
        """
        Load image as binary for handcrafted feature extraction
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Simple thresholding for feature extraction
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return binary
    
    def _save_user_profile(self, username: str, profile: Dict):
        """
        Save user profile to disk
        """
        profile_path = f"{self.profiles_path}/{username}_profile.pkl"
        
        try:
            with open(profile_path, 'wb') as f:
                pickle.dump(profile, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Also save a JSON summary for debugging
            json_summary = {
                'username': profile['username'],
                'enrollment_date': profile['enrollment_date'],
                'signature_count': profile['signature_count'],
                'thresholds': profile['thresholds'],
                'model_version': profile['model_version']
            }
            
            json_path = f"{self.profiles_path}/{username}_summary.json"
            with open(json_path, 'w') as f:
                json.dump(json_summary, f, indent=2)
            
            logger.info(f"Profile saved for user: {username}")
            
        except Exception as e:
            logger.error(f"Error saving profile for {username}: {e}")
            raise
    
    def _load_user_profile(self, username: str) -> Dict:
        """
        Load user profile from disk
        """
        profile_path = f"{self.profiles_path}/{username}_profile.pkl"
        
        if not os.path.exists(profile_path):
            return None
        
        try:
            with open(profile_path, 'rb') as f:
                profile = pickle.load(f)
            return profile
        except Exception as e:
            logger.error(f"Error loading profile for {username}: {e}")
            return None
    
    def _update_profile_statistics(self, username: str, verification_result: Dict):
        """
        Update profile statistics after each verification
        """
        profile = self._load_user_profile(username)
        if profile is None:
            return
        
        # Update usage statistics
        profile['verification_count'] += 1
        profile['last_verification'] = verification_result['timestamp']
        
        # Note: In real implementation, you'd need ground truth to update false rates
        # For now, just update verification count
        
        self._save_user_profile(username, profile)
    
    def _get_model_hash(self):
        """
        Get hash of current model for version tracking
        """
        import hashlib
        
        if self.feature_extractor is None:
            return "no_model"
        
        # Create hash from model weights
        weights_str = str([w.numpy().tobytes() for w in self.feature_extractor.weights])
        return hashlib.md5(weights_str.encode()).hexdigest()[:16]
    
    def get_user_statistics(self, username: str) -> Dict:
        """
        Get comprehensive statistics for a user
        """
        profile = self._load_user_profile(username)
        if profile is None:
            return {"error": "User profile not found"}
        
        stats = {
            'username': username,
            'enrollment_date': profile['enrollment_date'],
            'signature_count': profile['signature_count'],
            'verification_count': profile['verification_count'],
            'last_verification': profile.get('last_verification'),
            'thresholds': profile['thresholds'],
            'model_version': profile['model_version']
        }
        
        return stats
    
    def update_user_threshold(self, username: str, new_threshold: float):
        """
        Update user's verification threshold (for fine-tuning)
        """
        profile = self._load_user_profile(username)
        if profile is None:
            raise ValueError(f"User profile not found: {username}")
        
        # Validate threshold range
        if not (0.5 <= new_threshold <= 0.95):
            raise ValueError("Threshold must be between 0.5 and 0.95")
        
        profile['thresholds']['combined_threshold'] = new_threshold
        profile['thresholds']['manual_override'] = True
        profile['thresholds']['override_date'] = datetime.now().isoformat()
        
        self._save_user_profile(username, profile)
        logger.info(f"Updated threshold for {username} to {new_threshold}")
    
    def delete_user_profile(self, username: str):
        """
        Delete user profile and associated data
        """
        profile_path = f"{self.profiles_path}/{username}_profile.pkl"
        json_path = f"{self.profiles_path}/{username}_summary.json"
        
        try:
            if os.path.exists(profile_path):
                os.remove(profile_path)
            if os.path.exists(json_path):
                os.remove(json_path)
            
            logger.info(f"Profile deleted for user: {username}")
            return True
        except Exception as e:
            logger.error(f"Error deleting profile for {username}: {e}")
            return False
    
    def list_enrolled_users(self) -> List[str]:
        """
        List all enrolled users
        """
        try:
            profile_files = [f for f in os.listdir(self.profiles_path) if f.endswith('_profile.pkl')]
            usernames = [f.replace('_profile.pkl', '') for f in profile_files]
            return sorted(usernames)
        except Exception as e:
            logger.error(f"Error listing users: {e}")
            return []
    
    def evaluate_system_performance(self, test_data: List[Tuple[str, str, bool]]) -> Dict:
        """
        Evaluate system performance on test dataset
        
        Args:
            test_data: List of (username, signature_path, is_genuine) tuples
            
        Returns:
            Performance metrics dictionary
        """
        results = []
        predictions = []
        ground_truth = []
        
        for username, sig_path, is_genuine in test_data:
            try:
                result = self.verify_signature(username, sig_path)
                results.append(result)
                predictions.append(1 if result['verified'] else 0)
                ground_truth.append(1 if is_genuine else 0)
            except Exception as e:
                logger.error(f"Error verifying {username}/{sig_path}: {e}")
                continue
        
        if not predictions:
            return {"error": "No successful verifications"}
        
        # Compute metrics
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': float(accuracy_score(ground_truth, predictions)),
            'precision': float(precision_score(ground_truth, predictions, zero_division=0)),
            'recall': float(recall_score(ground_truth, predictions, zero_division=0)),
            'f1_score': float(f1_score(ground_truth, predictions, zero_division=0)),
            'total_tests': len(predictions),
            'true_positives': int(np.sum((predictions == 1) & (ground_truth == 1))),
            'true_negatives': int(np.sum((predictions == 0) & (ground_truth == 0))),
            'false_positives': int(np.sum((predictions == 1) & (ground_truth == 0))),
            'false_negatives': int(np.sum((predictions == 0) & (ground_truth == 1))),
        }
        
        # Compute rates
        metrics['false_acceptance_rate'] = metrics['false_positives'] / max(np.sum(ground_truth == 0), 1)
        metrics['false_rejection_rate'] = metrics['false_negatives'] / max(np.sum(ground_truth == 1), 1)
        
        return metrics


class ProfileManager:
    """
    Utility class for managing user profiles and batch operations
    """
    
    def __init__(self, verification_engine: SignatureVerificationEngine):
        self.engine = verification_engine
        self.profiles_path = verification_engine.profiles_path
    
    def batch_enroll_users(self, user_data: Dict[str, List[str]]) -> Dict:
        """
        Enroll multiple users in batch
        
        Args:
            user_data: {username: [list_of_signature_paths]}
            
        Returns:
            Enrollment results
        """
        results = {
            'successful': [],
            'failed': [],
            'total_users': len(user_data)
        }
        
        for username, signature_paths in user_data.items():
            try:
                profile = self.engine.create_user_profile(username, signature_paths)
                results['successful'].append({
                    'username': username,
                    'signature_count': profile['signature_count'],
                    'enrollment_date': profile['enrollment_date']
                })
                logger.info(f"Successfully enrolled user: {username}")
            except Exception as e:
                results['failed'].append({
                    'username': username,
                    'error': str(e)
                })
                logger.error(f"Failed to enroll user {username}: {e}")
        
        return results
    
    def export_user_profiles(self, output_path: str):
        """
        Export all user profiles for backup
        """
        users = self.engine.list_enrolled_users()
        export_data = {}
        
        for username in users:
            profile = self.engine._load_user_profile(username)
            if profile:
                # Remove large numpy arrays for export
                export_profile = {
                    'username': profile['username'],
                    'enrollment_date': profile['enrollment_date'],
                    'signature_count': profile['signature_count'],
                    'thresholds': profile['thresholds'],
                    'verification_count': profile['verification_count'],
                    'model_version': profile['model_version']
                }
                export_data[username] = export_profile
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} user profiles to {output_path}")
    
    def cleanup_old_profiles(self, days_threshold: int = 90):
        """
        Clean up profiles that haven't been used recently
        """
        from datetime import datetime, timedelta
        
        users = self.engine.list_enrolled_users()
        threshold_date = datetime.now() - timedelta(days=days_threshold)
        cleaned_count = 0
        
        for username in users:
            profile = self.engine._load_user_profile(username)
            if profile:
                last_verification = profile.get('last_verification')
                if last_verification:
                    last_date = datetime.fromisoformat(last_verification)
                    if last_date < threshold_date:
                        self.engine.delete_user_profile(username)
                        cleaned_count += 1
                        logger.info(f"Cleaned up inactive profile: {username}")
        
        logger.info(f"Cleaned up {cleaned_count} inactive profiles")
        return cleaned_count


# Advanced verification utilities
class VerificationUtils:
    """
    Utility functions for signature verification analysis
    """
    
    @staticmethod
    def analyze_signature_quality(image_path: str) -> Dict:
        """
        Analyze the quality of a signature image
        """
        processor = SignatureImageProcessor()
        
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {"error": "Cannot load image"}
        
        # Compute quality metrics
        quality_metrics = {
            'resolution': f"{image.shape[1]}x{image.shape[0]}",
            'aspect_ratio': image.shape[1] / image.shape[0],
            'contrast': float(image.std()),
            'sharpness': float(cv2.Laplacian(image, cv2.CV_64F).var()),
            'noise_level': VerificationUtils._estimate_noise_level(image),
            'signature_coverage': VerificationUtils._compute_signature_coverage(image),
            'stroke_continuity': VerificationUtils._analyze_stroke_continuity(image)
        }
        
        # Overall quality score (0-100)
        quality_score = VerificationUtils._compute_quality_score(quality_metrics)
        quality_metrics['overall_quality'] = quality_score
        quality_metrics['quality_grade'] = VerificationUtils._get_quality_grade(quality_score)
        
        return quality_metrics
    
    @staticmethod
    def _estimate_noise_level(image):
        """Estimate noise level in image"""
        # Use Laplacian variance as noise indicator
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        # Normalize to 0-1 scale
        noise_level = min(laplacian_var / 1000, 1.0)
        return float(noise_level)
    
    @staticmethod
    def _compute_signature_coverage(image):
        """Compute how much of the image is covered by signature"""
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        signature_pixels = np.sum(binary == 0)  # Dark pixels
        total_pixels = image.shape[0] * image.shape[1]
        coverage = signature_pixels / total_pixels
        return float(coverage)
    
    @staticmethod
    def _analyze_stroke_continuity(image):
        """Analyze stroke continuity and connectedness"""
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        if num_labels <= 1:  # Only background
            return 0.0
        
        # Analyze component sizes
        component_sizes = stats[1:, cv2.CC_STAT_AREA]  # Exclude background
        main_component_size = np.max(component_sizes)
        total_signature_area = np.sum(component_sizes)
        
        # Continuity score based on main component dominance
        continuity = main_component_size / total_signature_area if total_signature_area > 0 else 0
        return float(continuity)
    
    @staticmethod
    def _compute_quality_score(metrics):
        """Compute overall quality score from individual metrics"""
        weights = {
            'contrast': 0.25,
            'sharpness': 0.25,
            'signature_coverage': 0.20,
            'stroke_continuity': 0.20,
            'noise_level': -0.10  # Negative weight (lower noise is better)
        }
        
        # Normalize metrics to 0-1 scale
        normalized_metrics = {
            'contrast': min(metrics['contrast'] / 50, 1.0),
            'sharpness': min(metrics['sharpness'] / 500, 1.0),
            'signature_coverage': min(metrics['signature_coverage'] * 10, 1.0),
            'stroke_continuity': metrics['stroke_continuity'],
            'noise_level': metrics['noise_level']
        }
        
        # Compute weighted score
        score = sum(weights[key] * normalized_metrics[key] for key in weights.keys())
        
        # Convert to 0-100 scale
        return max(0, min(100, score * 100))
    
    @staticmethod
    def _get_quality_grade(score):
        """Convert quality score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'


# Example usage and testing
if __name__ == "__main__":
    # Initialize verification engine
    engine = SignatureVerificationEngine()
    
    # Test user enrollment
    test_username = "test_user"
    test_signatures = [
        "../data/users/test_user/train/signature_1.png",
        "../data/users/test_user/train/signature_2.png",
        "../data/users/test_user/train/signature_3.png",
        "../data/users/test_user/train/signature_4.png",
        "../data/users/test_user/train/signature_5.png"
    ]
    
    try:
        # Create user profile
        profile = engine.create_user_profile(test_username, test_signatures)
        print(f"Profile created successfully for {test_username}")
        print(f"Signature count: {profile['signature_count']}")
        print(f"Thresholds: {profile['thresholds']}")
        
        # Test verification
        test_signature = "../data/users/test_user/test/test_signature.png"
        if os.path.exists(test_signature):
            result = engine.verify_signature(test_username, test_signature)
            print(f"Verification result: {result}")
        
        # Get user statistics
        stats = engine.get_user_statistics(test_username)
        print(f"User statistics: {stats}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    # Test quality analysis
    if os.path.exists(test_signatures[0]):
        quality = VerificationUtils.analyze_signature_quality(test_signatures[0])
        print(f"Signature quality analysis: {quality}")