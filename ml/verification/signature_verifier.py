"""
Advanced Signature Verification System
Uses the sophisticated Siamese network with proper thresholding
Updated to work with ml/training/data directory structure and SiameseNetwork class
"""

import os
import pickle
import json
import numpy as np
from datetime import datetime
import sys
from pathlib import Path
import tensorflow as tf
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import your updated SiameseNetwork
try:
    from ml.models.siamese_network import SiameseNetwork
except ImportError:
    print("Warning: Could not import SiameseNetwork. Some features may not work.")
    SiameseNetwork = None

class SignatureVerifier:
    def __init__(self):
        self.target_size = (224, 224)
        self.siamese_model = None
        self.siamese_network = None
        self.feature_extractor = None
        self.model_metadata = None
        
        # Updated directory paths to match your structure
        self.profiles_dir = "ml/training/data/profiles"
        self.models_dir = "ml/training/data/models"
        self.users_dir = "ml/training/data/users"
        
        # Create directories
        os.makedirs(self.profiles_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.users_dir, exist_ok=True)
    
    def load_or_create_model(self):
        """Load the Siamese model from training directory"""
        # Try to load the best performing model first
        model_paths = [
            f"{self.models_dir}/siamese_signature_model.h5",
            f"{self.models_dir}/siamese_signature_model_underperforming.h5",
            f"{self.models_dir}/best_siamese_model.h5"
        ]
        
        metadata_path = f"{self.models_dir}/model_metadata.json"
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    # Load model with minimal custom objects
                    self.siamese_model = tf.keras.models.load_model(
                        model_path, 
                        compile=False  # Don't compile to avoid custom loss issues
                    )
                    
                    # Load metadata
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            self.model_metadata = json.load(f)
                    
                    # Try to load backbone separately
                    backbone_paths = [
                        f"{self.models_dir}/signature_backbone.h5",
                        f"{self.models_dir}/signature_backbone_underperforming.h5"
                    ]
                    
                    for backbone_path in backbone_paths:
                        if os.path.exists(backbone_path):
                            try:
                                self.feature_extractor = tf.keras.models.load_model(
                                    backbone_path, compile=False
                                )
                                break
                            except:
                                continue
                    
                    print(f"Siamese model loaded successfully from {model_path}")
                    if self.model_metadata:
                        print(f"Model AUC: {self.model_metadata.get('validation_auc', 'Unknown')}")
                        print(f"Model Status: {self.model_metadata.get('status', 'Unknown')}")
                    
                    return True
                    
                except Exception as e:
                    print(f"Failed to load model from {model_path}: {e}")
                    continue
        
        # If no trained model found, try to create SiameseNetwork instance
        if SiameseNetwork is not None:
            try:
                print("No trained model found. Initializing SiameseNetwork...")
                self.siamese_network = SiameseNetwork(
                    input_shape=(224, 224, 3),
                    models_dir=self.models_dir
                )
                print("SiameseNetwork initialized. Train a model first for full functionality.")
                return False
            except Exception as e:
                print(f"Failed to initialize SiameseNetwork: {e}")
        
        print("No trained model found and SiameseNetwork not available.")
        print("Please run training first or check model paths.")
        return False
    
    def _preprocess_signature(self, image_path):
        """Simple preprocessing to match training pipeline"""
        try:
            # Load and preprocess image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            raise ValueError(f"Failed to preprocess {image_path}: {e}")
    
    def _extract_handcrafted_features(self, image_path):
        """Extract simple handcrafted features"""
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Resize for consistency
            img = cv2.resize(img, self.target_size)
            
            # Simple features
            features = []
            
            # 1. Image statistics
            features.extend([
                np.mean(img),
                np.std(img),
                np.median(img),
                np.min(img),
                np.max(img)
            ])
            
            # 2. Histogram features (8 bins)
            hist = cv2.calcHist([img], [0], None, [8], [0, 256])
            features.extend(hist.flatten() / np.sum(hist))
            
            # 3. Gradient features
            grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            
            features.extend([
                np.mean(np.abs(grad_x)),
                np.mean(np.abs(grad_y)),
                np.std(grad_x),
                np.std(grad_y)
            ])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Warning: Failed to extract handcrafted features: {e}")
            # Return zeros if extraction fails
            return np.zeros(17)  # 5 + 8 + 4 features
    
    def enroll_user(self, username, signature_paths):
        """
        Enroll user with multiple signatures
        """
        if len(signature_paths) < 3:
            raise ValueError("Need at least 3 signatures for enrollment")
        
        print(f"Enrolling user: {username}")
        print(f"Processing {len(signature_paths)} signatures...")
        
        # Process signatures
        raw_images = []
        cnn_features = []
        handcrafted_features = []
        valid_paths = []
        
        for i, sig_path in enumerate(signature_paths):
            try:
                # Preprocess image
                processed_img = self._preprocess_signature(sig_path)
                raw_images.append(processed_img)
                
                # Extract CNN features if model available
                if self.feature_extractor is not None:
                    features = self.feature_extractor.predict(
                        np.expand_dims(processed_img, axis=0), 
                        verbose=0
                    )
                    cnn_features.append(features.flatten())
                
                # Extract handcrafted features
                hc_features = self._extract_handcrafted_features(sig_path)
                handcrafted_features.append(hc_features)
                
                valid_paths.append(str(sig_path))
                print(f"  Processed signature {i+1}/{len(signature_paths)}")
                
            except Exception as e:
                print(f"  Failed to process {sig_path}: {e}")
                continue
        
        if len(valid_paths) < 3:
            raise ValueError(f"Only {len(valid_paths)} signatures processed successfully. Need at least 3.")
        
        # Calculate thresholds
        threshold = self._calculate_thresholds(raw_images, cnn_features, handcrafted_features)
        
        # Create profile
        profile = {
            'username': username,
            'enrollment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'signature_count': len(valid_paths),
            'signature_paths': valid_paths,
            
            # Store processed data
            'raw_images': raw_images,
            'cnn_features': cnn_features if cnn_features else None,
            'handcrafted_features': handcrafted_features,
            
            # Statistics
            'handcrafted_mean': np.mean(handcrafted_features, axis=0).tolist(),
            'handcrafted_std': np.std(handcrafted_features, axis=0).tolist(),
            
            # Thresholds
            'siamese_threshold': threshold['siamese'],
            'cnn_threshold': threshold['cnn'],
            'handcrafted_threshold': threshold['handcrafted'],
            'combined_threshold': threshold['combined'],
            
            # Model info
            'model_metadata': self.model_metadata
        }
        
        # Add CNN statistics if available
        if cnn_features:
            profile['cnn_mean'] = np.mean(cnn_features, axis=0).tolist()
            profile['cnn_std'] = np.std(cnn_features, axis=0).tolist()
        
        # Save profile
        profile_path = f"{self.profiles_dir}/{username}_profile.pkl"
        with open(profile_path, 'wb') as f:
            pickle.dump(profile, f)
        
        print(f"User {username} enrolled successfully!")
        print(f"  - {profile['signature_count']} signatures processed")
        print(f"  - Siamese threshold: {profile['siamese_threshold']:.3f}")
        print(f"  - Profile saved to: {profile_path}")
        
        return profile
    
    def _calculate_thresholds(self, raw_images, cnn_features, handcrafted_features):
        """Calculate verification thresholds"""
        
        thresholds = {
            'siamese': 0.7,
            'cnn': 0.7, 
            'handcrafted': 0.7,
            'combined': 0.7
        }
        
        # Siamese threshold
        if self.siamese_model is not None and len(raw_images) >= 2:
            similarities = []
            
            for i in range(len(raw_images)):
                for j in range(i+1, len(raw_images)):
                    try:
                        score = self.siamese_model.predict([
                            np.expand_dims(raw_images[i], axis=0),
                            np.expand_dims(raw_images[j], axis=0)
                        ], verbose=0)[0][0]
                        similarities.append(score)
                    except:
                        continue
            
            if similarities:
                mean_sim = np.mean(similarities)
                std_sim = np.std(similarities)
                thresholds['siamese'] = max(0.3, min(0.8, mean_sim - 2*std_sim))
        
        # CNN threshold
        if cnn_features and len(cnn_features) >= 2:
            from sklearn.metrics.pairwise import cosine_similarity
            
            similarities = []
            for i in range(len(cnn_features)):
                for j in range(i+1, len(cnn_features)):
                    sim = cosine_similarity([cnn_features[i]], [cnn_features[j]])[0][0]
                    similarities.append(sim)
            
            if similarities:
                mean_sim = np.mean(similarities)
                std_sim = np.std(similarities)
                thresholds['cnn'] = max(0.4, min(0.9, mean_sim - 1.5*std_sim))
        
        # Handcrafted threshold
        if len(handcrafted_features) >= 2:
            from sklearn.metrics.pairwise import cosine_similarity
            
            similarities = []
            for i in range(len(handcrafted_features)):
                for j in range(i+1, len(handcrafted_features)):
                    sim = cosine_similarity([handcrafted_features[i]], [handcrafted_features[j]])[0][0]
                    if not np.isnan(sim):
                        similarities.append(sim)
            
            if similarities:
                mean_sim = np.mean(similarities)
                std_sim = np.std(similarities)
                thresholds['handcrafted'] = max(0.3, min(0.8, mean_sim - 1.5*std_sim))
        
        # Combined threshold
        thresholds['combined'] = min(
            thresholds['siamese'],
            thresholds['cnn'], 
            thresholds['handcrafted']
        )
        
        return thresholds
    
    def verify_signature(self, username, test_signature_path):
        """
        Verify signature against enrolled user
        """
        # Load user profile
        profile_path = f"{self.profiles_dir}/{username}_profile.pkl"
        
        if not os.path.exists(profile_path):
            return {
                'verified': False,
                'error': f'User {username} not enrolled',
                'confidence': 0.0,
                'username': username
            }
        
        with open(profile_path, 'rb') as f:
            profile = pickle.load(f)
        
        try:
            # Process test signature
            test_processed = self._preprocess_signature(test_signature_path)
            test_hc_features = self._extract_handcrafted_features(test_signature_path)
            
            # Initialize scores
            siamese_score = 0.0
            cnn_score = 0.0
            handcrafted_score = 0.0
            
            # 1. Siamese network verification
            if self.siamese_model is not None and 'raw_images' in profile:
                siamese_scores = []
                
                for enrolled_img in profile['raw_images']:
                    try:
                        score = self.siamese_model.predict([
                            np.expand_dims(test_processed, axis=0),
                            np.expand_dims(enrolled_img, axis=0)
                        ], verbose=0)[0][0]
                        siamese_scores.append(score)
                    except Exception as e:
                        print(f"Siamese prediction error: {e}")
                        continue
                
                if siamese_scores:
                    siamese_score = max(siamese_scores)
            
            # 2. CNN feature verification
            if self.feature_extractor is not None and profile.get('cnn_features') is not None:
                try:
                    test_cnn_features = self.feature_extractor.predict(
                        np.expand_dims(test_processed, axis=0), 
                        verbose=0
                    ).flatten()
                    
                    from sklearn.metrics.pairwise import cosine_similarity
                    cnn_scores = []
                    
                    for enrolled_cnn in profile['cnn_features']:
                        sim = cosine_similarity([test_cnn_features], [enrolled_cnn])[0][0]
                        cnn_scores.append(sim)
                    
                    if cnn_scores:
                        cnn_score = max(cnn_scores)
                        
                except Exception as e:
                    print(f"CNN feature error: {e}")
            
            # 3. Handcrafted features verification
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                hc_scores = []
                
                for enrolled_hc in profile['handcrafted_features']:
                    sim = cosine_similarity([test_hc_features], [enrolled_hc])[0][0]
                    if not np.isnan(sim):
                        hc_scores.append(sim)
                
                if hc_scores:
                    handcrafted_score = max(hc_scores)
                    
            except Exception as e:
                print(f"Handcrafted feature error: {e}")
            
            # Decision making
            weights = {
                'siamese': 0.6,
                'cnn': 0.3,
                'handcrafted': 0.1
            }
            
            # Calculate weighted score
            total_score = (
                weights['siamese'] * siamese_score +
                weights['cnn'] * cnn_score +
                weights['handcrafted'] * handcrafted_score
            )
            
            # Individual verifications
            siamese_verified = siamese_score > profile['siamese_threshold']
            cnn_verified = cnn_score > profile['cnn_threshold']
            handcrafted_verified = handcrafted_score > profile['handcrafted_threshold']
            
            # Final decision - require at least 2 methods to agree
            agreement_count = sum([siamese_verified, cnn_verified, handcrafted_verified])
            final_verified = agreement_count >= 2
            
            # If Siamese model is available, it should pass for final verification
            if self.siamese_model is not None:
                final_verified = siamese_verified and agreement_count >= 2
            
            # Confidence calculation
            confidence = min(100, total_score * 100)
            
            # Adjust confidence based on agreement
            if agreement_count >= 2:
                confidence *= 1.0
            elif agreement_count == 1:
                confidence *= 0.7
            else:
                confidence *= 0.3
            
            result = {
                'verified': bool(final_verified),
                'confidence': float(confidence),
                'total_score': float(total_score),
                
                # Individual scores
                'siamese_score': float(siamese_score),
                'cnn_score': float(cnn_score),
                'handcrafted_score': float(handcrafted_score),
                
                # Individual verifications
                'siamese_verified': bool(siamese_verified),
                'cnn_verified': bool(cnn_verified),
                'handcrafted_verified': bool(handcrafted_verified),
                
                # Thresholds
                'siamese_threshold': float(profile['siamese_threshold']),
                'cnn_threshold': float(profile['cnn_threshold']),
                'handcrafted_threshold': float(profile['handcrafted_threshold']),
                
                # Metadata
                'username': username,
                'test_image': str(test_signature_path),
                'verification_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'methods_agreement': int(agreement_count)
            }
            
            print(f"Verification Result for {username}:")
            print(f"  - Final Verified: {final_verified}")
            print(f"  - Confidence: {confidence:.1f}%")
            print(f"  - Siamese: {siamese_score:.3f} ({'✓' if siamese_verified else '✗'})")
            print(f"  - CNN: {cnn_score:.3f} ({'✓' if cnn_verified else '✗'})")
            print(f"  - Handcrafted: {handcrafted_score:.3f} ({'✓' if handcrafted_verified else '✗'})")
            print(f"  - Agreement: {agreement_count}/3")
            
            return result
            
        except Exception as e:
            return {
                'verified': False,
                'error': f'Verification failed: {str(e)}',
                'confidence': 0.0,
                'username': username,
                'verification_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def get_enrolled_users(self):
        """Get list of enrolled users"""
        users = []
        if os.path.exists(self.profiles_dir):
            for filename in os.listdir(self.profiles_dir):
                if filename.endswith('_profile.pkl'):
                    username = filename.replace('_profile.pkl', '')
                    users.append(username)
        return users
    
    def get_user_info(self, username):
        """Get information about enrolled user"""
        profile_path = f"{self.profiles_dir}/{username}_profile.pkl"
        
        if not os.path.exists(profile_path):
            return None
        
        try:
            with open(profile_path, 'rb') as f:
                profile = pickle.load(f)
            
            return {
                'username': profile['username'],
                'enrollment_date': profile['enrollment_date'],
                'signature_count': profile['signature_count'],
                'siamese_threshold': profile['siamese_threshold'],
                'cnn_threshold': profile['cnn_threshold'],
                'handcrafted_threshold': profile['handcrafted_threshold'],
                'combined_threshold': profile['combined_threshold']
            }
        except Exception as e:
            print(f"Error loading user info: {e}")
            return None