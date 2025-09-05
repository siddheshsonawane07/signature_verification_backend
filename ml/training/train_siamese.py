"""
High-Accuracy Training Script for Signature Verification
Focused on maximizing performance with advanced techniques
"""

import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import cv2
import json
from sklearn.metrics import classification_report, confusion_matrix
import albumentations as A

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.models.siamese_network import SiameseNetwork
from ml.preprocessing.image_preprocessor import ImageProcessor

class HighAccuracyTrainer:
    def __init__(self):
        self.image_processor = ImageProcessor(target_size=(224, 224))
        self.siamese_net = SiameseNetwork()
        self.data_dir = "data/users"
        self.models_dir = "data/models"
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        random.seed(42)
        
        # Initialize advanced augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self):
        """Create advanced augmentation pipeline using Albumentations"""
        try:
            return A.Compose([
                A.Rotate(limit=5, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.6),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=3, p=0.5),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.2),
                A.ElasticTransform(alpha=1, sigma=20, alpha_affine=10, p=0.1)
            ])
        except ImportError:
            print("Albumentations not available, using basic augmentation")
            return None
    
    def load_data_enhanced(self):
        """Load data with enhanced preprocessing for maximum quality"""
        print("Loading signature data with enhanced preprocessing...")
        
        user_signatures = defaultdict(list)
        users_dir = Path(self.data_dir)
        
        if not users_dir.exists():
            raise ValueError(f"Directory {self.data_dir} not found!")
        
        total_signatures = 0
        for user_folder in users_dir.iterdir():
            if user_folder.is_dir():
                train_dir = user_folder / "train"
                if train_dir.exists():
                    image_files = []
                    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                        image_files.extend(train_dir.glob(ext))
                    
                    if len(image_files) >= 3:
                        processed_images = []
                        
                        for img_path in image_files:
                            try:
                                # Advanced preprocessing
                                img = self._preprocess_image_advanced(str(img_path))
                                if img is not None:
                                    processed_images.append(img)
                                
                            except Exception as e:
                                print(f"  Failed to process {img_path}: {e}")
                                continue
                        
                        if len(processed_images) >= 3:
                            user_signatures[user_folder.name] = processed_images
                            total_signatures += len(processed_images)
                            print(f"  Loaded {user_folder.name}: {len(processed_images)} signatures")
        
        print(f"Total: {total_signatures} signatures from {len(user_signatures)} users")
        return user_signatures
    
    def _preprocess_image_advanced(self, image_path):
        """Advanced image preprocessing for better feature extraction"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Advanced contrast enhancement
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge back
            img = cv2.merge([l, a, b])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
            
            # Bilateral filter for noise reduction while preserving edges
            img = cv2.bilateralFilter(img, 9, 75, 75)
            
            # Resize with high-quality interpolation
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Histogram equalization per channel
            for i in range(3):
                channel = img[:, :, i]
                channel = cv2.equalizeHist((channel * 255).astype(np.uint8)) / 255.0
                img[:, :, i] = channel
            
            # Slight sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.1
            img_sharp = cv2.filter2D(img, -1, kernel)
            img = 0.8 * img + 0.2 * img_sharp
            img = np.clip(img, 0, 1)
            
            return img
            
        except Exception as e:
            print(f"Advanced preprocessing error: {e}")
            return None
    
    def create_comprehensive_pairs(self, user_signatures):
        """Create comprehensive training pairs with strategic selection"""
        print("Creating comprehensive training pairs...")
        
        users = list(user_signatures.keys())
        pairs = []
        labels = []
        
        # Calculate optimal number of pairs
        total_images = sum(len(sigs) for sigs in user_signatures.values())
        
        # More aggressive pair creation for better learning
        max_pairs_per_class = min(50, total_images)  # Increased from 30
        
        # Create positive pairs with maximum diversity
        positive_pairs = 0
        for user in users:
            images = user_signatures[user]
            pairs_from_user = 0
            max_from_user = max(5, max_pairs_per_class // len(users))
            
            # Create all possible pairs for this user (if not too many)
            if len(images) <= 8:
                for i in range(len(images)):
                    for j in range(i + 1, len(images)):
                        if positive_pairs < max_pairs_per_class and pairs_from_user < max_from_user:
                            pairs.append([images[i], images[j]])
                            labels.append(1)
                            positive_pairs += 1
                            pairs_from_user += 1
            else:
                # For users with many signatures, sample strategically
                for _ in range(max_from_user):
                    if positive_pairs < max_pairs_per_class:
                        i, j = random.sample(range(len(images)), 2)
                        pairs.append([images[i], images[j]])
                        labels.append(1)
                        positive_pairs += 1
                        pairs_from_user += 1
            
            if positive_pairs >= max_pairs_per_class:
                break
        
        # Create challenging negative pairs
        negative_pairs = 0
        user_combinations = []
        
        # Generate all possible user combinations
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user_combinations.append((users[i], users[j]))
        
        # Ensure good coverage of negative pairs
        while negative_pairs < positive_pairs:
            if user_combinations:
                user1, user2 = random.choice(user_combinations)
            else:
                user1, user2 = random.sample(users, 2)
            
            img1 = random.choice(user_signatures[user1])
            img2 = random.choice(user_signatures[user2])
            
            pairs.append([img1, img2])
            labels.append(0)
            negative_pairs += 1
        
        print(f"Created {len(pairs)} comprehensive pairs: {positive_pairs} positive, {negative_pairs} negative")
        return pairs, labels
    
    def advanced_augmentation(self, pairs, labels):
        """Advanced data augmentation for robust training"""
        print("Applying advanced augmentation...")
        
        augmented_pairs = []
        augmented_labels = []
        
        # Add original pairs
        augmented_pairs.extend(pairs)
        augmented_labels.extend(labels)
        
        # Multiple augmentation rounds
        augmentation_rounds = 3  # Increased augmentation
        
        for round_num in range(augmentation_rounds):
            print(f"  Augmentation round {round_num + 1}/{augmentation_rounds}")
            
            for (img1, img2), label in zip(pairs, labels):
                try:
                    if self.augmentation_pipeline:
                        # Use Albumentations if available
                        aug1 = self.augmentation_pipeline(image=(img1 * 255).astype(np.uint8))['image'] / 255.0
                        aug2 = self.augmentation_pipeline(image=(img2 * 255).astype(np.uint8))['image'] / 255.0
                    else:
                        # Fallback to manual augmentation
                        aug1 = self._manual_augment(img1)
                        aug2 = self._manual_augment(img2)
                    
                    augmented_pairs.append([aug1, aug2])
                    augmented_labels.append(label)
                    
                except Exception as e:
                    # If augmentation fails, use original
                    augmented_pairs.append([img1, img2])
                    augmented_labels.append(label)
        
        print(f"Augmented dataset: {len(augmented_pairs)} pairs")
        return augmented_pairs, augmented_labels
    
    def _manual_augment(self, image):
        """Manual augmentation fallback"""
        img = image.copy()
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-3, 3)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 1)
        
        # Random noise
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.01, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        return img.astype(np.float32)
    
    def create_advanced_model(self):
        """Create advanced Siamese model with state-of-the-art techniques"""
        
        def create_backbone():
            """Create enhanced backbone network"""
            inputs = tf.keras.layers.Input(shape=(224, 224, 3))
            
            # Initial convolution with larger kernel
            x = tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
            
            # Residual blocks with increasing complexity
            filters = [64, 128, 256, 512]
            
            for i, f in enumerate(filters):
                # Multiple residual blocks per stage
                for j in range(2 if i < 2 else 3):
                    # Residual block
                    shortcut = x
                    
                    # Main path
                    x = tf.keras.layers.Conv2D(f, (3, 3), padding='same')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.ReLU()(x)
                    x = tf.keras.layers.Conv2D(f, (3, 3), padding='same')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    
                    # Dimension matching for shortcut
                    if shortcut.shape[-1] != f:
                        shortcut = tf.keras.layers.Conv2D(f, (1, 1), padding='same')(shortcut)
                        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
                    
                    # Add shortcut
                    x = tf.keras.layers.Add()([x, shortcut])
                    x = tf.keras.layers.ReLU()(x)
                
                # Downsampling between stages
                if i < len(filters) - 1:
                    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                    x = tf.keras.layers.Dropout(0.1)(x)
            
            # Global feature extraction with attention
            # Self-attention mechanism
            attention = tf.keras.layers.GlobalAveragePooling2D()(x)
            attention = tf.keras.layers.Dense(x.shape[-1] // 8, activation='relu')(attention)
            attention = tf.keras.layers.Dense(x.shape[-1], activation='sigmoid')(attention)
            attention = tf.keras.layers.Reshape((1, 1, x.shape[-1]))(attention)
            x = tf.keras.layers.Multiply()([x, attention])
            
            # Global pooling
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            
            # Dense layers with progressive size reduction
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.4)(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            features = tf.keras.layers.Dense(128, activation='relu', name='features')(x)
            
            return tf.keras.Model(inputs, features, name='advanced_backbone')
        
        # Create backbone
        backbone = create_backbone()
        
        # Siamese inputs
        input_a = tf.keras.layers.Input(shape=(224, 224, 3), name='input_a')
        input_b = tf.keras.layers.Input(shape=(224, 224, 3), name='input_b')
        
        # Extract features
        features_a = backbone(input_a)
        features_b = backbone(input_b)
        
        # Multiple distance measures
        l1_distance = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([features_a, features_b])
        l2_distance = tf.keras.layers.Lambda(lambda x: tf.square(x[0] - x[1]))([features_a, features_b])
        
        # Cosine similarity
        def cosine_similarity(vectors):
            x, y = vectors
            x_norm = tf.nn.l2_normalize(x, axis=1)
            y_norm = tf.nn.l2_normalize(y, axis=1)
            return tf.reduce_sum(x_norm * y_norm, axis=1, keepdims=True)
        
        cosine_sim = tf.keras.layers.Lambda(cosine_similarity)([features_a, features_b])
        
        # Element-wise operations
        element_mult = tf.keras.layers.Multiply()([features_a, features_b])
        element_add = tf.keras.layers.Add()([features_a, features_b])
        
        # Concatenate all similarity measures
        combined = tf.keras.layers.Concatenate()([
            l1_distance, l2_distance, cosine_sim, element_mult, element_add
        ])
        
        # Advanced decision network
        x = tf.keras.layers.Dense(512, activation='relu')(combined)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        # Output layer
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='similarity')(x)
        
        # Create final model
        model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
        
        return model, backbone
    
    def train_high_accuracy_model(self, pairs, labels):
        """Train model with focus on maximum accuracy"""
        print("\\nTraining high-accuracy Siamese network...")
        
        # Prepare data
        left_images = np.array([pair[0] for pair in pairs], dtype=np.float32)
        right_images = np.array([pair[1] for pair in pairs], dtype=np.float32)
        labels_array = np.array(labels, dtype=np.float32)
        
        print(f"Total training pairs: {len(labels_array)}")
        print(f"Class distribution: {np.bincount(labels_array.astype(int))}")
        
        # Create model
        model, backbone = self.create_advanced_model()
        
        # Advanced optimizer with learning rate scheduling
        initial_lr = 0.001
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=initial_lr,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999
        )
        
        # Compile with focal loss for better class balancing
        model.compile(
            optimizer=optimizer,
            loss=self._focal_loss,
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.F1Score(name='f1')
            ]
        )
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(labels_array), 
            y=labels_array
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        print(f"Class weights: {class_weight_dict}")
        
        # Advanced callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=15,  # Increased patience
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=8,
                min_lr=1e-8,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f"{self.models_dir}/best_high_accuracy_model.h5",
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: initial_lr * (0.95 ** epoch),
                verbose=0
            )
        ]
        
        # Train with validation split
        history = model.fit(
            [left_images, right_images], labels_array,
            batch_size=32,  # Larger batch for stability
            epochs=50,     # More epochs for convergence
            validation_split=0.2,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Final evaluation on validation set
        val_split_idx = int(len(labels_array) * 0.8)
        val_left = left_images[val_split_idx:]
        val_right = right_images[val_split_idx:]
        val_labels = labels_array[val_split_idx:]
        
        val_results = model.evaluate([val_left, val_right], val_labels, verbose=0)
        val_loss, val_acc, val_precision, val_recall, val_auc, val_f1 = val_results
        
        print(f"\\nFinal Validation Results:")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation Precision: {val_precision:.4f}")
        print(f"Validation Recall: {val_recall:.4f}")
        print(f"Validation AUC: {val_auc:.4f}")
        print(f"Validation F1-Score: {val_f1:.4f}")
        
        # Detailed analysis
        val_predictions = model.predict([val_left, val_right], verbose=0)
        val_pred_binary = (val_predictions > 0.5).astype(int).flatten()
        
        # Calculate detailed metrics
        tp = np.sum((val_pred_binary == 1) & (val_labels == 1))
        tn = np.sum((val_pred_binary == 0) & (val_labels == 0))
        fp = np.sum((val_pred_binary == 1) & (val_labels == 0))
        fn = np.sum((val_pred_binary == 0) & (val_labels == 1))
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print(f"\\nDetailed Analysis:")
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"False Acceptance Rate: {far:.4f}")
        print(f"False Rejection Rate: {frr:.4f}")
        
        # Save model if performance is good
        if val_auc > 0.8 and val_acc > 0.75:
            model.save(f"{self.models_dir}/high_accuracy_siamese_model.h5")
            backbone.save(f"{self.models_dir}/high_accuracy_backbone.h5")
            
            metadata = {
                'validation_accuracy': float(val_acc),
                'validation_auc': float(val_auc),
                'validation_precision': float(val_precision),
                'validation_recall': float(val_recall),
                'validation_f1': float(val_f1),
                'false_acceptance_rate': float(far),
                'false_rejection_rate': float(frr),
                'model_type': 'high_accuracy_siamese',
                'training_pairs': len(labels_array),
                'epochs_trained': len(history.history['loss'])
            }
            
            with open(f"{self.models_dir}/high_accuracy_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\\nHigh-accuracy model saved with AUC: {val_auc:.4f}")
            return True, history, metadata
        else:
            print(f"\\nModel performance needs improvement (AUC: {val_auc:.4f}, Acc: {val_acc:.4f})")
            return False, history, None
    
    def _focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2.0):
        """Focal loss for handling class imbalance"""
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        
        focal_weight = alpha_t * tf.pow(1 - pt, gamma)
        focal_loss = -focal_weight * tf.math.log(pt)
        
        return tf.reduce_mean(focal_loss)
    
    def plot_comprehensive_results(self, history):
        """Create comprehensive training plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC
        axes[0, 2].plot(history.history['auc'], label='Training', linewidth=2)
        axes[0, 2].plot(history.history['val_auc'], label='Validation', linewidth=2)
        axes[0, 2].set_title('Model AUC', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('AUC')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training', linewidth=2)
        axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Training', linewidth=2)
        axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # F1-Score
        axes[1, 2].plot(history.history['f1'], label='Training', linewidth=2)
        axes[1, 2].plot(history.history['val_f1'], label='Validation', linewidth=2)
        axes[1, 2].set_title('F1-Score', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('F1-Score')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.models_dir}/high_accuracy_training_results.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_training(self):
        """Complete high-accuracy training pipeline"""
        print("="*70)
        print("HIGH-ACCURACY SIGNATURE VERIFICATION - TRAINING")
        print("="*70)
        
        try:
            # Load data with enhanced preprocessing
            user_signatures = self.load_data_enhanced()
            
            if len(user_signatures) < 2:
                print("Error: Need at least 2 users for training!")
                return False
            
            # Create comprehensive pairs
            pairs, labels = self.create_comprehensive_pairs(user_signatures)
            
            if len(pairs) < 20:
                print("Error: Not enough pairs for training!")
                return False
            
            # Advanced augmentation
            augmented_pairs, augmented_labels = self.advanced_augmentation(pairs, labels)
            
            # High-accuracy training
            success, history, metadata = self.train_high_accuracy_model(augmented_pairs, augmented_labels)
            
            # Comprehensive plotting
            if history:
                self.plot_comprehensive_results(history)
            
            if success:
                print("\\n" + "="*70)
                print("HIGH-ACCURACY TRAINING COMPLETED SUCCESSFULLY!")
                print("="*70)
                print(f"Final Validation AUC: {metadata['validation_auc']:.4f}")
                print(f"Final Validation Accuracy: {metadata['validation_accuracy']:.4f}")
                print(f"Final Validation F1-Score: {metadata['validation_f1']:.4f}")
                print(f"False Acceptance Rate: {metadata['false_acceptance_rate']:.4f}")
                print(f"False Rejection Rate: {metadata['false_rejection_rate']:.4f}")
                print(f"Training completed in {metadata['epochs_trained']} epochs")
                print("\\nHigh-accuracy model is ready for deployment!")
            else:
                print("\\nTraining completed but performance targets not met.")
                print("Consider adding more diverse signature data or adjusting parameters.")
            
            return success
                
        except Exception as e:
            print(f"\\nHigh-accuracy training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main training function"""
    trainer = HighAccuracyTrainer()
    success = trainer.run_training()
    
    if not success:
        print("\\nTroubleshooting tips:")
        print("1. Ensure you have at least 3 users with 5+ signatures each")
        print("2. Verify signatures show genuine variation between users")
        print("3. Check image quality and proper file formats")
        print("4. Consider collecting more diverse signature samples")
        print("5. Install albumentations for better augmentation: pip install albumentations")

if __name__ == "__main__":
    main()