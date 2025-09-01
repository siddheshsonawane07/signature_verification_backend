"""
Advanced Signature Verification System
Uses the sophisticated Siamese network with proper thresholding
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

from ml.preprocessing.image_preprocessor import ImageProcessor

class SignatureVerifier:
    def __init__(self):
        self.image_processor = ImageProcessor(target_size=(224, 224))
        self.siamese_model = None
        self.feature_extractor = None
        self.model_metadata = None
        self.profiles_dir = "data/profiles"
        self.models_dir = "data/models"
        
        # Create directories
        os.makedirs(self.profiles_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_or_create_model(self):
        """Load the advanced Siamese model"""
        model_path = f"{self.models_dir}/siamese_model.h5"
        metadata_path = f"{self.models_dir}/model_metadata.json"
        
        if os.path.exists(model_path):
            try:
                # Custom objects for loading
                custom_objects = {
                    'focal_loss': self._focal_loss,
                    'contrastive_loss': self._contrastive_loss
                }
                
                self.siamese_model = tf.keras.models.load_model(
                    model_path, custom_objects=custom_objects, compile=False
                )
                
                # Load metadata
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.model_metadata = json.load(f)
                
                # Try to load feature extractor separately
                feature_path = f"{self.models_dir}/feature_extractor.h5"
                if os.path.exists(feature_path):
                    self.feature_extractor = tf.keras.models.load_model(feature_path)
                
                print(f"Advanced Siamese model loaded successfully")
                print(f"Model AUC: {self.model_metadata.get('test_auc', 'Unknown'):.4f}")
                print(f"False Acceptance Rate: {self.model_metadata.get('false_acceptance_rate', 'Unknown'):.4f}")
                
                return True
                
            except Exception as e:
                print(f"Failed to load advanced model: {e}")
                print("Will use correlation-based fallback")
                return False
        else:
            print("No trained model found. Please run advanced training first.")
            print("Will use correlation-based verification.")
            return False
    
    def _focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2.0):
        """Focal loss for loading model"""
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow((1.0 - p_t), gamma)
        loss = alpha_factor * modulating_factor * ce
        return tf.reduce_mean(loss)
    
    def _contrastive_loss(self, y_true, y_pred, margin=1.0):
        """Contrastive loss for loading model"""
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    
    def enroll_user(self, username, signature_paths):
        """
        Advanced user enrollment with multiple feature types
        """
        if len(signature_paths) < 3:
            raise ValueError("Need at least 3 signatures for enrollment")
        
        print(f"Enrolling user: {username}")
        print(f"Processing {len(signature_paths)} signatures...")
        
        # Process signatures with advanced preprocessing
        cnn_features = []
        handcrafted_features = []
        raw_images = []
        valid_paths = []
        
        for i, sig_path in enumerate(signature_paths):
            try:
                # CNN preprocessing
                processed_img = self.image_processor.preprocess_signature(sig_path)
                raw_images.append(processed_img[0])  # Store for CNN
                
                # Extract CNN features if model is available
                if self.feature_extractor is not None:
                    cnn_feature = self.feature_extractor.predict(processed_img, verbose=0)
                    cnn_features.append(cnn_feature.flatten())
                
                # Extract handcrafted features
                hc_features = self.image_processor.extract_handcrafted_features(sig_path)
                handcrafted_features.append(hc_features)
                
                valid_paths.append(sig_path)
                print(f"  Processed signature {i+1}/{len(signature_paths)}")
                
            except Exception as e:
                print(f"  Failed to process {sig_path}: {e}")
                continue
        
        if len(valid_paths) < 3:
            raise ValueError(f"Only {len(valid_paths)} signatures processed successfully. Need at least 3.")
        
        # Calculate sophisticated threshold
        threshold = self._calculate_advanced_threshold(
            raw_images, cnn_features, handcrafted_features
        )
        
        # Create comprehensive profile
        profile = {
            'username': username,
            'enrollment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'signature_count': len(valid_paths),
            'signature_paths': valid_paths,
            
            # Store raw images for Siamese network
            'raw_images': raw_images,
            
            # CNN features (if available)
            'cnn_features': cnn_features if cnn_features else None,
            'cnn_mean': np.mean(cnn_features, axis=0) if cnn_features else None,
            'cnn_std': np.std(cnn_features, axis=0) if cnn_features else None,
            
            # Handcrafted features
            'handcrafted_features': handcrafted_features,
            'handcrafted_mean': np.mean(handcrafted_features, axis=0),
            'handcrafted_std': np.std(handcrafted_features, axis=0),
            
            # Thresholds
            'siamese_threshold': threshold['siamese'],
            'cnn_threshold': threshold['cnn'],
            'handcrafted_threshold': threshold['handcrafted'],
            'combined_threshold': threshold['combined'],
            
            # Model info
            'model_metadata': self.model_metadata
        }
        
        # Save profile
        profile_path = f"{self.profiles_dir}/{username}_profile.pkl"
        with open(profile_path, 'wb') as f:
            pickle.dump(profile, f)
        
        print(f"User {username} enrolled successfully!")
        print(f"  - {profile['signature_count']} signatures processed")
        print(f"  - Siamese threshold: {profile['siamese_threshold']:.3f}")
        print(f"  - Combined threshold: {profile['combined_threshold']:.3f}")
        print(f"  - Profile saved to: {profile_path}")
        
        return profile
    
    def _calculate_advanced_threshold(self, raw_images, cnn_features, handcrafted_features):
        """Calculate sophisticated thresholds using multiple methods"""
        
        thresholds = {
            'siamese': 0.7,  # Default
            'cnn': 0.7,
            'handcrafted': 0.7,
            'combined': 0.7
        }
        
        # Siamese threshold - if model is available
        if self.siamese_model is not None and len(raw_images) >= 2:
            siamese_similarities = []
            
            for i in range(len(raw_images)):
                for j in range(i+1, len(raw_images)):
                    try:
                        # Get similarity from Siamese network
                        sim_score = self.siamese_model.predict([
                            np.expand_dims(raw_images[i], axis=0),
                            np.expand_dims(raw_images[j], axis=0)
                        ], verbose=0)[0][0]
                        siamese_similarities.append(sim_score)
                    except:
                        continue
            
            if siamese_similarities:
                mean_sim = np.mean(siamese_similarities)
                std_sim = np.std(siamese_similarities)
                # Conservative threshold: mean - 2*std, but not too low
                thresholds['siamese'] = max(0.3, min(0.8, mean_sim - 2*std_sim))
        
        # CNN feature threshold
        if cnn_features and len(cnn_features) >= 2:
            from sklearn.metrics.pairwise import cosine_similarity
            
            cnn_similarities = []
            for i in range(len(cnn_features)):
                for j in range(i+1, len(cnn_features)):
                    sim = cosine_similarity([cnn_features[i]], [cnn_features[j]])[0][0]
                    cnn_similarities.append(sim)
            
            if cnn_similarities:
                mean_sim = np.mean(cnn_similarities)
                std_sim = np.std(cnn_similarities)
                thresholds['cnn'] = max(0.4, min(0.9, mean_sim - 1.5*std_sim))
        
        # Handcrafted features threshold
        if len(handcrafted_features) >= 2:
            from sklearn.metrics.pairwise import cosine_similarity
            
            hc_similarities = []
            for i in range(len(handcrafted_features)):
                for j in range(i+1, len(handcrafted_features)):
                    sim = cosine_similarity([handcrafted_features[i]], [handcrafted_features[j]])[0][0]
                    if not np.isnan(sim):
                        hc_similarities.append(sim)
            
            if hc_similarities:
                mean_sim = np.mean(hc_similarities)
                std_sim = np.std(hc_similarities)
                thresholds['handcrafted'] = max(0.3, min(0.8, mean_sim - 1.5*std_sim))
        
        # Combined threshold (conservative approach)
        thresholds['combined'] = min(
            thresholds['siamese'], 
            thresholds['cnn'], 
            thresholds['handcrafted']
        )
        
        return thresholds
    
    def verify_signature(self, username, test_signature_path):
        """
        Advanced signature verification using multiple methods
        """
        # Load user profile
        profile_path = f"{self.profiles_dir}/{username}_profile.pkl"
        
        if not os.path.exists(profile_path):
            return {
                'verified': False,
                'error': f'User {username} not enrolled',
                'confidence': 0.0
            }
        
        with open(profile_path, 'rb') as f:
            profile = pickle.load(f)
        
        try:
            # Process test signature
            test_processed = self.image_processor.preprocess_signature(test_signature_path)
            test_hc_features = self.image_processor.extract_handcrafted_features(test_signature_path)
            
            # Initialize scores
            siamese_score = 0.0
            cnn_score = 0.0
            handcrafted_score = 0.0
            
            # 1. Siamese network verification (most important)
            if self.siamese_model is not None and 'raw_images' in profile:
                siamese_scores = []
                
                for enrolled_img in profile['raw_images']:
                    try:
                        score = self.siamese_model.predict([
                            test_processed,
                            np.expand_dims(enrolled_img, axis=0)
                        ], verbose=0)[0][0]
                        siamese_scores.append(score)
                    except:
                        continue
                
                if siamese_scores:
                    siamese_score = max(siamese_scores)
            
            # 2. CNN feature verification (if available)
            if self.feature_extractor is not None and profile['cnn_features'] is not None:
                test_cnn_features = self.feature_extractor.predict(test_processed, verbose=0).flatten()
                
                from sklearn.metrics.pairwise import cosine_similarity
                cnn_scores = []
                
                for enrolled_cnn in profile['cnn_features']:
                    sim = cosine_similarity([test_cnn_features], [enrolled_cnn])[0][0]
                    cnn_scores.append(sim)
                
                if cnn_scores:
                    cnn_score = max(cnn_scores)
            
            # 3. Handcrafted features verification
            from sklearn.metrics.pairwise import cosine_similarity
            hc_scores = []
            
            for enrolled_hc in profile['handcrafted_features']:
                sim = cosine_similarity([test_hc_features], [enrolled_hc])[0][0]
                if not np.isnan(sim):
                    hc_scores.append(sim)
            
            if hc_scores:
                handcrafted_score = max(hc_scores)
            
            # Decision making - weighted combination
            weights = {
                'siamese': 0.6,      # Highest weight to advanced model
                'cnn': 0.3,          # Medium weight to CNN features
                'handcrafted': 0.1   # Lowest weight to handcrafted
            }
            
            # Calculate weighted score
            total_score = (
                weights['siamese'] * siamese_score +
                weights['cnn'] * cnn_score +
                weights['handcrafted'] * handcrafted_score
            )
            
            # Individual thresholds
            siamese_verified = siamese_score > profile['siamese_threshold']
            cnn_verified = cnn_score > profile['cnn_threshold']
            handcrafted_verified = handcrafted_score > profile['handcrafted_threshold']
            
            # Final decision - require Siamese to pass AND at least one other method
            final_verified = siamese_verified and (cnn_verified or handcrafted_verified)
            
            # Confidence calculation
            confidence = min(100, total_score * 100)
            
            # Adjust confidence based on agreement
            agreement_count = sum([siamese_verified, cnn_verified, handcrafted_verified])
            if agreement_count >= 2:
                confidence *= 1.0  # Full confidence
            elif agreement_count == 1:
                confidence *= 0.7  # Reduced confidence
            else:
                confidence *= 0.3  # Very low confidence
            
            result = {
                'verified': bool(final_verified),
                'confidence': float(confidence),
                'total_score': float(total_score),
                
                # Individual method scores
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
                'test_image': test_signature_path,
                'verification_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'methods_agreement': int(agreement_count)
            }
            
            print(f"Advanced Verification Result for {username}:")
            print(f"  - Final Verified: {final_verified}")
            print(f"  - Confidence: {confidence:.1f}%")
            print(f"  - Siamese: {siamese_score:.3f} (threshold: {profile['siamese_threshold']:.3f}) {'✓' if siamese_verified else '✗'}")
            print(f"  - CNN: {cnn_score:.3f} (threshold: {profile['cnn_threshold']:.3f}) {'✓' if cnn_verified else '✗'}")
            print(f"  - Handcrafted: {handcrafted_score:.3f} (threshold: {profile['handcrafted_threshold']:.3f}) {'✓' if handcrafted_verified else '✗'}")
            print(f"  - Methods Agreement: {agreement_count}/3")
            
            return result
            
        except Exception as e:
            return {
                'verified': False,
                'error': f'Verification failed: {str(e)}',
                'confidence': 0.0,
                'username': username
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
        """Get detailed information about an enrolled user"""
        profile_path = f"{self.profiles_dir}/{username}_profile.pkl"
        
        if not os.path.exists(profile_path):
            return None
        
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