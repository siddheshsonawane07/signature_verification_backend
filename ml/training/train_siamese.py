"""
Updated SiameseTrainer to use simplified network for better performance
"""

import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import cv2
import json
from datetime import datetime

# Import the simplified SiameseNetwork
from ml.models.siamese_network import SiameseNetwork

class SiameseTrainer:
    """
    Professional Siamese Network Trainer for Signature Verification
    Updated to use simplified network architecture
    """
    
    def __init__(self, 
                 data_dir: str = "ml/training/data/users",
                 models_dir: str = "ml/training/data/models",
                 target_size: tuple = (224, 224),
                 batch_size: int = 16,  # Smaller batch for stability
                 epochs: int = 10):
        """
        Initialize the Siamese Network Trainer
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.target_size = target_size
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Create necessary directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize attributes for storing results
        self.training_history = None
        self.model = None
        self.backbone = None
        self.training_metadata = {}
        self.training_results = {}
        
        print(f"Siamese Trainer initialized:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Models directory: {self.models_dir}")
        print(f"  Target image size: {target_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Max epochs: {epochs}")
    
    def load_signature_data(self):
        """
        Load and preprocess signature data from user directories
        """
        print("Loading signature data with advanced preprocessing...")
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {self.data_dir} not found!")
        
        user_signatures = defaultdict(list)
        total_signatures = 0
        
        # Iterate through user directories
        for user_folder in self.data_dir.iterdir():
            if not user_folder.is_dir():
                continue
                
            train_dir = user_folder / "train"
            if not train_dir.exists():
                continue
            
            # Find all image files with supported extensions
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            image_files = []
            for ext in image_extensions:
                image_files.extend(train_dir.glob(ext))
            
            # Process images for this user
            if len(image_files) >= 3:  # Minimum required signatures per user
                processed_images = []
                
                for img_path in image_files:
                    try:
                        processed_img = self._preprocess_signature(str(img_path))
                        if processed_img is not None:
                            processed_images.append(processed_img)
                    except Exception as e:
                        print(f"  Warning: Failed to process {img_path}: {e}")
                        continue
                
                # Store user data if sufficient signatures processed
                if len(processed_images) >= 3:
                    user_signatures[user_folder.name] = processed_images
                    total_signatures += len(processed_images)
                    print(f"  ✓ User {user_folder.name}: {len(processed_images)} signatures")
                else:
                    print(f"  ✗ User {user_folder.name}: Insufficient valid signatures")
        
        print(f"\nData loading complete:")
        print(f"  Total users: {len(user_signatures)}")
        print(f"  Total signatures: {total_signatures}")
        if len(user_signatures) > 0:
            print(f"  Average signatures per user: {total_signatures/len(user_signatures):.1f}")
        
        if len(user_signatures) < 2:
            raise ValueError("Need at least 2 users with 3+ signatures each for training!")
        
        return dict(user_signatures)
    
    def _preprocess_signature(self, image_path: str):
        """
        Simple but effective preprocessing for signature images
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Simple contrast enhancement
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
            
            # Resize to target size
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize to [0, 1] range
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            print(f"Error in preprocessing {image_path}: {e}")
            return None
    
    def create_training_pairs(self, user_signatures):
        """
        Create balanced training pairs for Siamese network
        """
        print("Creating training pairs...")
        
        users = list(user_signatures.keys())
        pairs = []
        labels = []
        
        # Target number of pairs
        max_pairs_per_class = 50  # Reduced for stability
        
        # Generate positive pairs (genuine-genuine)
        positive_count = 0
        print("  Generating positive pairs (genuine-genuine)...")
        
        for user in users:
            images = user_signatures[user]
            user_pairs = min(max_pairs_per_class // len(users), 10)  # Max 10 per user
            
            # Create pairs within user signatures
            pairs_created = 0
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    if positive_count < max_pairs_per_class and pairs_created < user_pairs:
                        pairs.append([images[i], images[j]])
                        labels.append(1)
                        positive_count += 1
                        pairs_created += 1
                    else:
                        break
                if pairs_created >= user_pairs:
                    break
            
            if positive_count >= max_pairs_per_class:
                break
        
        # Generate negative pairs (different users)
        negative_count = 0
        print("  Generating negative pairs (genuine-forged)...")
        
        while negative_count < positive_count and len(users) >= 2:
            user1, user2 = random.sample(users, 2)
            img1 = random.choice(user_signatures[user1])
            img2 = random.choice(user_signatures[user2])
            
            pairs.append([img1, img2])
            labels.append(0)
            negative_count += 1
        
        print(f"  Created {len(pairs)} total pairs:")
        print(f"    Positive pairs: {positive_count}")
        print(f"    Negative pairs: {negative_count}")
        print(f"    Class balance: {positive_count/(positive_count+negative_count):.3f}")
        
        return pairs, labels
    
    def apply_data_augmentation(self, pairs, labels):
        """
        Apply minimal data augmentation to avoid overfitting
        """
        print("Applying light data augmentation...")
        
        augmented_pairs = list(pairs)
        augmented_labels = list(labels)
        
        # Only one round of light augmentation
        for (img1, img2), label in zip(pairs, labels):
            try:
                # Light augmentation
                aug_img1 = self._light_augmentation(img1)
                aug_img2 = self._light_augmentation(img2)
                
                augmented_pairs.append([aug_img1, aug_img2])
                augmented_labels.append(label)
                
            except Exception as e:
                # If augmentation fails, use original images
                augmented_pairs.append([img1, img2])
                augmented_labels.append(label)
        
        print(f"  Dataset augmented: {len(pairs)} → {len(augmented_pairs)} pairs")
        
        return augmented_pairs, augmented_labels
    
    def _light_augmentation(self, image):
        """
        Apply very light augmentation to avoid overfitting
        """
        img = image.copy()
        
        # Random brightness (very light)
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.9, 1.1)
            img = np.clip(img * brightness_factor, 0, 1)
        
        # Random noise (very light)
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.01, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        return img.astype(np.float32)
    
    def train_model(self, pairs, labels):
        """
        Train the simplified Siamese network
        """
        print("\nTraining simplified Siamese network...")

        # Prepare training data
        left_images = np.array([pair[0] for pair in pairs], dtype=np.float32)
        right_images = np.array([pair[1] for pair in pairs], dtype=np.float32)
        labels_array = np.array(labels, dtype=np.float32)

        print(f"Training data prepared:")
        print(f"  Total pairs: {len(labels_array):,}")
        print(f"  Positive pairs: {np.sum(labels_array):,.0f}")
        print(f"  Negative pairs: {len(labels_array) - np.sum(labels_array):,.0f}")
        print(f"  Class ratio: {np.mean(labels_array):.3f}")

        # Create simplified model
        siamese_net = SiameseNetwork(
            input_shape=(*self.target_size, 3),
            models_dir=str(self.models_dir)
        )
        model = siamese_net.build_siamese_model()
        self.model = model
        self.backbone = siamese_net.get_feature_extractor()

        print(f"Model parameters: {model.count_params():,}")

        # Simple optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # Lower learning rate

        # Compile model with simple loss
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',  # Simple binary crossentropy
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
            ]
        )

        # Balanced class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels_array),
            y=labels_array
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        print(f"Class weights: {class_weight_dict}")

        # Simple callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train
        print("\nStarting model training...")
        history = model.fit(
            [left_images, right_images], labels_array,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )

        self.training_history = history

        # Evaluation
        val_split_idx = int(len(labels_array) * 0.8)
        val_left = left_images[val_split_idx:]
        val_right = right_images[val_split_idx:]
        val_labels = labels_array[val_split_idx:]

        if len(val_labels) > 0:
            val_results = model.evaluate([val_left, val_right], val_labels, verbose=0)
            val_loss, val_acc, val_precision, val_recall, val_auc, val_binary_acc = val_results

            # Predictions for detailed analysis
            val_predictions = model.predict([val_left, val_right], verbose=0)
            val_pred_binary = (val_predictions > 0.5).astype(int).flatten()

            # Confusion matrix
            tp = np.sum((val_pred_binary == 1) & (val_labels == 1))
            tn = np.sum((val_pred_binary == 0) & (val_labels == 0))
            fp = np.sum((val_pred_binary == 1) & (val_labels == 0))
            fn = np.sum((val_pred_binary == 0) & (val_labels == 1))

            # Metrics
            precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
            val_f1 = 2 * (precision_manual * recall_manual) / (precision_manual + recall_manual) if (precision_manual + recall_manual) > 0 else 0
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0

            # Store results
            self.training_results = {
                'validation_accuracy': val_acc,
                'validation_precision': val_precision,
                'validation_recall': val_recall,
                'validation_auc': val_auc,
                'validation_f1_score': val_f1,
                'false_acceptance_rate': far,
                'false_rejection_rate': frr
            }

            # Print results
            print(f"\n{'='*60}")
            print(f"TRAINING COMPLETED - PERFORMANCE SUMMARY")
            print(f"{'='*60}")
            print(f"Validation Accuracy:     {val_acc:.4f}")
            print(f"Validation Precision:    {val_precision:.4f}")
            print(f"Validation Recall:       {val_recall:.4f}")
            print(f"Validation AUC:          {val_auc:.4f}")
            print(f"Validation F1-Score:     {val_f1:.4f}")
            print(f"False Acceptance Rate:   {far:.4f}")
            print(f"False Rejection Rate:    {frr:.4f}")
            print(f"Training Epochs:         {len(history.history['loss'])}")

            # Quality assessment
            model_quality = val_auc > 0.75 and val_acc > 0.70 and far < 0.20

            # Save model
            if model_quality:
                model_path = self.models_dir / "siamese_signature_model.h5"
                backbone_path = self.models_dir / "signature_backbone.h5" 
                status = "good"
            else:
                model_path = self.models_dir / "siamese_signature_model_underperforming.h5"
                backbone_path = self.models_dir / "signature_backbone_underperforming.h5"
                status = "underperforming"

            model.save(str(model_path))
            self.backbone.save(str(backbone_path))

            # Save metadata
            metadata = {
                'model_type': 'simplified_siamese_signature_verification',
                'status': status,
                'validation_accuracy': float(val_acc),
                'validation_precision': float(val_precision),
                'validation_recall': float(val_recall),
                'validation_auc': float(val_auc),
                'validation_f1_score': float(val_f1),
                'false_acceptance_rate': float(far),
                'false_rejection_rate': float(frr),
                'training_parameters': {
                    'total_pairs': len(labels_array),
                    'batch_size': self.batch_size,
                    'epochs_trained': len(history.history['loss']),
                    'learning_rate': 0.0005
                },
                'model_architecture': {
                    'backbone_params': int(self.backbone.count_params()),
                    'total_params': int(model.count_params()),
                    'architecture': 'MobileNetV2 + Simple Head'
                }
            }

            self.training_metadata = metadata

            with open(self.models_dir / "model_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"\n✓ Model saved at {model_path} (status: {status})")
            return model_quality, history, metadata

        else:
            print(f"\n⚠ No validation data available")
            return False, history, None

    def generate_research_report(self):
        """Generate training report"""
        print("Generating research report...")
        
        try:
            report_path = self.models_dir / "training_report.txt"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(report_path, "w") as f:
                f.write("=" * 80 + "\n")
                f.write("SIMPLIFIED SIAMESE NETWORK TRAINING REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {timestamp}\n\n")
                
                f.write("ARCHITECTURE CHANGES\n")
                f.write("-" * 40 + "\n")
                f.write("- Simplified to MobileNetV2 + basic head\n")
                f.write("- Reduced complexity to prevent overfitting\n")
                f.write("- Lower learning rate for stability\n")
                f.write("- Light data augmentation only\n")
                f.write("- Binary crossentropy loss (no focal loss)\n\n")
                
                if hasattr(self, 'training_results'):
                    f.write("PERFORMANCE RESULTS\n")
                    f.write("-" * 40 + "\n")
                    results = self.training_results
                    f.write(f"Validation Accuracy:     {results.get('validation_accuracy', 0):.4f}\n")
                    f.write(f"Validation AUC:          {results.get('validation_auc', 0):.4f}\n")
                    f.write(f"Validation F1-Score:     {results.get('validation_f1_score', 0):.4f}\n")
                    f.write(f"False Acceptance Rate:   {results.get('false_acceptance_rate', 0):.4f}\n")
                    f.write(f"False Rejection Rate:    {results.get('false_rejection_rate', 0):.4f}\n\n")
                
                f.write("NEXT STEPS\n")
                f.write("-" * 40 + "\n")
                f.write("- If performance is still poor, collect more data\n")
                f.write("- Consider fine-tuning MobileNet layers\n")
                f.write("- Experiment with different thresholds\n")
                f.write("- Test with real signature verification scenarios\n")
                
            print(f"  ✓ Report saved to {report_path}")
            return report_path
            
        except Exception as e:
            print(f"  ✗ Error generating report: {e}")
            return None

    def run_complete_training_pipeline(self):
        """Execute the complete simplified training pipeline"""
        print(f"\n{'='*80}")
        print(f"SIMPLIFIED SIAMESE NETWORK TRAINING PIPELINE")
        print(f"{'='*80}")
        
        try:
            # Step 1: Load data
            print(f"\n[1/4] Loading signature data...")
            user_signatures = self.load_signature_data()
            
            # Step 2: Create pairs
            print(f"\n[2/4] Creating training pairs...")
            pairs, labels = self.create_training_pairs(user_signatures)
            
            if len(pairs) < 10:
                raise ValueError("Insufficient training pairs generated!")
            
            # Step 3: Light augmentation
            print(f"\n[3/4] Applying light augmentation...")
            augmented_pairs, augmented_labels = self.apply_data_augmentation(pairs, labels)
            
            # Step 4: Train model
            print(f"\n[4/4] Training simplified model...")
            success, history, metadata = self.train_model(augmented_pairs, augmented_labels)
            
            # Generate report
            self.generate_research_report()
            
            # Final summary
            if success and metadata:
                print(f"\n{'='*60}")
                print(f"TRAINING COMPLETED SUCCESSFULLY!")
                print(f"{'='*60}")
                print(f"✓ Simplified architecture performed well")
                print(f"✓ Model ready for signature verification")
                return True
            else:
                print(f"\n{'='*60}")
                print(f"TRAINING COMPLETED WITH ISSUES")
                print(f"{'='*60}")
                print(f"⚠ Model needs more data or tuning")
                return False
                
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main training function"""
    print("Initializing Simplified Siamese Network Trainer...")
    
    trainer = SiameseTrainer(
        data_dir="ml/training/data/users",
        models_dir="ml/training/data/models", 
        target_size=(224, 224),
        batch_size=16,
        epochs=10
    )
    
    success = trainer.run_complete_training_pipeline()
    
    if success:
        print(f"\n🎉 Training completed successfully!")
    else:
        print(f"\n⚠ Training had issues but model was saved")
    
    return success


if __name__ == "__main__":
    # Configure TensorFlow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU acceleration enabled")
        except RuntimeError as e:
            print(f"GPU error: {e}")
    else:
        print("⚠ No GPU detected, using CPU")
    
    success = main()