"""
Simplified and Compatible Signature Verification System
Works seamlessly with the simplified SiameseNetwork and training pipeline
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

# Import simplified preprocessing
def simple_preprocess_signature(image_path, target_size=(224, 224)):
    """Simple preprocessing that matches training pipeline"""
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Simple contrast enhancement (same as training)
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        
        # Resize to target size
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
        
    except Exception as e:
        raise ValueError(f"Failed to preprocess {image_path}: {e}")

def extract_simple_features(image_path, target_size=(224, 224)):
    """Extract simple handcrafted features"""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize for consistency
        img = cv2.resize(img, target_size)
        
        # Basic statistical features
        features = [
            np.mean(img),
            np.std(img),
            np.median(img),
            np.min(img),
            np.max(img),
            np.percentile(img, 25),
            np.percentile(img, 75)
        ]
        
        # Histogram features (4 bins for simplicity)
        hist = cv2.calcHist([img], [0], None, [4], [0, 256])
        features.extend(hist.flatten() / np.sum(hist))
        
        # Gradient features
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        features.extend([
            np.mean(np.abs(grad_x)),
            np.mean(np.abs(grad_y))
        ])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Warning: Failed to extract features: {e}")
        return np.zeros(13)  # 7 + 4 + 2 features

class SignatureVerifier:
    def __init__(self):
        self.target_size = (224, 224)
        self.siamese_model = None
        self.feature_extractor = None
        self.model_metadata = None
        
        # Directory paths matching the training structure
        self.profiles_dir = "ml/training/data/profiles"
        self.models_dir = "ml/training/data/models"
        self.users_dir = "ml/training/data/users"
        
        # Create directories
        os.makedirs(self.profiles_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.users_dir, exist_ok=True)
    
    def load_or_create_model(self):
        """Load the trained Siamese model"""
        # Try different model paths in order of preference
        model_paths = [
            f"{self.models_dir}/siamese_signature_model.h5",
            f"{self.models_dir}/siamese_signature_model_underperforming.h5",
            f"{self.models_dir}/best_siamese_model.h5"
        ]
        
        metadata_path = f"{self.models_dir}/model_metadata.json"
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    # Load model without compilation to avoid custom loss issues
                    self.siamese_model = tf.keras.models.load_model(
                        model_path, 
                        compile=False
                    )
                    
                    # Load metadata if available
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            self.model_metadata = json.load(f)
                    
                    # Try to load backbone/feature extractor
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
                    
                    print(f"Siamese model loaded from {model_path}")
                    if self.model_metadata:
                        print(f"Model AUC: {self.model_metadata.get('validation_auc', 'Unknown'):.3f}")
                        print(f"Model Status: {self.model_metadata.get('status', 'Unknown')}")
                    
                    return True
                    
                except Exception as e:
                    print(f"Failed to load model from {model_path}: {e}")
                    continue
        
        print("No trained model found. Please run training first.")
        return False
    
    def enroll_user(self, username, signature_paths):
        """Enroll user with simplified processing"""
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
                # Preprocess image (same as training)
                processed_img = simple_preprocess_signature(sig_path, self.target_size)
                raw_images.append(processed_img)
                
                # Extract CNN features if available
                if self.feature_extractor is not None:
                    features = self.feature_extractor.predict(
                        np.expand_dims(processed_img, axis=0), 
                        verbose=0
                    )
                    cnn_features.append(features.flatten())
                
                # Extract simple handcrafted features
                hc_features = extract_simple_features(sig_path, self.target_size)
                handcrafted_features.append(hc_features)
                
                valid_paths.append(str(sig_path))
                print(f"  Processed signature {i+1}/{len(signature_paths)}")
                
            except Exception as e:
                print(f"  Failed to process {sig_path}: {e}")
                continue
        
        if len(valid_paths) < 3:
            raise ValueError(f"Only {len(valid_paths)} signatures processed. Need at least 3.")
        
        # Calculate thresholds
        thresholds = self._calculate_thresholds(raw_images, cnn_features, handcrafted_features)
        
        # Create user profile
        profile = {
            'username': username,
            'enrollment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'signature_count': len(valid_paths),
            'signature_paths': valid_paths,
            'raw_images': raw_images,
            'cnn_features': cnn_features if cnn_features else None,
            'handcrafted_features': handcrafted_features,
            'siamese_threshold': thresholds['siamese'],
            'cnn_threshold': thresholds['cnn'],
            'handcrafted_threshold': thresholds['handcrafted'],
            'model_metadata': self.model_metadata
        }
        
        # Add feature statistics
        if cnn_features:
            profile['cnn_mean'] = np.mean(cnn_features, axis=0).tolist()
            profile['cnn_std'] = np.std(cnn_features, axis=0).tolist()
        
        profile['handcrafted_mean'] = np.mean(handcrafted_features, axis=0).tolist()
        profile['handcrafted_std'] = np.std(handcrafted_features, axis=0).tolist()
        
        # Save profile
        profile_path = f"{self.profiles_dir}/{username}_profile.pkl"
        with open(profile_path, 'wb') as f:
            pickle.dump(profile, f)
        
        print(f"User {username} enrolled successfully!")
        print(f"  - {profile['signature_count']} signatures processed")
        print(f"  - Siamese threshold: {profile['siamese_threshold']:.3f}")
        print(f"  - Profile saved")
        
        return profile
    
    def _calculate_thresholds(self, raw_images, cnn_features, handcrafted_features):
        """Calculate verification thresholds"""
        thresholds = {
            'siamese': 0.6,  # Slightly lower default for simplified model
            'cnn': 0.7,
            'handcrafted': 0.6
        }
        
        # Siamese threshold from intra-user similarities
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
                # Conservative threshold
                thresholds['siamese'] = max(0.3, min(0.8, mean_sim - 1.5*std_sim))
        
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
                thresholds['cnn'] = max(0.4, min(0.9, mean_sim - 1.0*std_sim))
        
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
                thresholds['handcrafted'] = max(0.3, min(0.8, mean_sim - 1.0*std_sim))
        
        return thresholds
    
    def verify_signature(self, username, test_signature_path):
        """Verify signature using simplified approach"""
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
            test_processed = simple_preprocess_signature(test_signature_path, self.target_size)
            test_hc_features = extract_simple_features(test_signature_path, self.target_size)
            
            # Initialize scores
            siamese_score = 0.0
            cnn_score = 0.0
            handcrafted_score = 0.0
            
            # 1. Siamese network verification (primary method)
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
                    siamese_score = max(siamese_scores)  # Best match
            
            # 2. CNN features verification (if available)
            if self.feature_extractor is not None and profile.get('cnn_features'):
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
            
            # 3. Handcrafted features verification (fallback)
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
            
            # Decision making with simplified logic
            siamese_verified = siamese_score > profile['siamese_threshold']
            cnn_verified = cnn_score > profile['cnn_threshold']
            handcrafted_verified = handcrafted_score > profile['handcrafted_threshold']
            
            # Primary decision based on Siamese network
            if self.siamese_model is not None:
                final_verified = siamese_verified
                confidence = siamese_score * 100
            else:
                # Fallback to majority vote
                votes = [cnn_verified, handcrafted_verified]
                final_verified = sum(votes) >= 1
                confidence = max(cnn_score, handcrafted_score) * 100
            
            # Boost confidence if multiple methods agree
            agreement_count = sum([siamese_verified, cnn_verified, handcrafted_verified])
            if agreement_count >= 2:
                confidence = min(100, confidence * 1.2)
            
            result = {
                'verified': bool(final_verified),
                'confidence': float(confidence),
                'siamese_score': float(siamese_score),
                'cnn_score': float(cnn_score),
                'handcrafted_score': float(handcrafted_score),
                'siamese_verified': bool(siamese_verified),
                'cnn_verified': bool(cnn_verified),
                'handcrafted_verified': bool(handcrafted_verified),
                'siamese_threshold': float(profile['siamese_threshold']),
                'cnn_threshold': float(profile['cnn_threshold']),
                'handcrafted_threshold': float(profile['handcrafted_threshold']),
                'username': username,
                'test_image': str(test_signature_path),
                'verification_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'methods_agreement': int(agreement_count)
            }
            
            print(f"Verification Result for {username}:")
            print(f"  - Verified: {final_verified}")
            print(f"  - Confidence: {confidence:.1f}%")
            print(f"  - Siamese: {siamese_score:.3f} ({'✓' if siamese_verified else '✗'})")
            if cnn_score > 0:
                print(f"  - CNN: {cnn_score:.3f} ({'✓' if cnn_verified else '✗'})")
            print(f"  - Handcrafted: {handcrafted_score:.3f} ({'✓' if handcrafted_verified else '✗'})")
            
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
                'handcrafted_threshold': profile['handcrafted_threshold']
            }
        except Exception as e:
            print(f"Error loading user info: {e}")
            return None