"""
Fast Training Script for Signature Verification
Optimized for speed while maintaining performance
"""

import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import cv2
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.models.siamese_network import SiameseNetwork
from ml.preprocessing.image_preprocessor import ImageProcessor

# Enable mixed precision for faster training (only if you have GPU)
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

class FastTrainer:
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
    
    def load_data_optimized(self):
        """Load data with optimized preprocessing"""
        print("Loading signature data with fast preprocessing...")
        
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
                                # Simple preprocessing - just resize and normalize
                                img = cv2.imread(str(img_path))
                                if img is not None:
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    img = cv2.resize(img, (224, 224))
                                    img = img.astype(np.float32) / 255.0
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
    
    def create_balanced_pairs(self, user_signatures):
        """Create balanced positive and negative pairs efficiently"""
        print("Creating balanced training pairs...")
        
        users = list(user_signatures.keys())
        pairs = []
        labels = []
        
        # Calculate max pairs to keep dataset manageable
        total_images = sum(len(sigs) for sigs in user_signatures.values())
        max_pairs_per_class = min(30, total_images // 2)  # Limit to 60 total pairs max
        
        # Create positive pairs (same user) - select diverse pairs
        positive_pairs = 0
        for user in users:
            images = user_signatures[user]
            pairs_from_user = 0
            max_from_user = max_pairs_per_class // len(users)
            
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    if positive_pairs < max_pairs_per_class and pairs_from_user < max_from_user:
                        pairs.append([images[i], images[j]])
                        labels.append(1)
                        positive_pairs += 1
                        pairs_from_user += 1
                    if positive_pairs >= max_pairs_per_class:
                        break
                if positive_pairs >= max_pairs_per_class:
                    break
        
        # Create exactly equal negative pairs
        negative_pairs = 0
        while negative_pairs < positive_pairs:
            user1, user2 = random.sample(users, 2)
            img1 = random.choice(user_signatures[user1])
            img2 = random.choice(user_signatures[user2])
            
            pairs.append([img1, img2])
            labels.append(0)
            negative_pairs += 1
        
        print(f"Created {len(pairs)} balanced pairs: {positive_pairs} positive, {negative_pairs} negative")
        return pairs, labels
    
    def minimal_augmentation(self, pairs, labels):
        """Light augmentation to double dataset size"""
        print("Applying minimal augmentation...")
        
        augmented_pairs = list(pairs)  # Copy original pairs
        augmented_labels = list(labels)
        
        # Add one augmented version of each pair
        for (img1, img2), label in zip(pairs, labels):
            # Simple rotation augmentation
            angle1 = random.uniform(-2, 2)
            angle2 = random.uniform(-2, 2)
            
            img1_aug = self._rotate_image(img1, angle1)
            img2_aug = self._rotate_image(img2, angle2)
            
            augmented_pairs.append([img1_aug, img2_aug])
            augmented_labels.append(label)
        
        print(f"Augmented dataset: {len(augmented_pairs)} pairs")
        return augmented_pairs, augmented_labels
    
    def _rotate_image(self, image, angle):
        """Fast image rotation"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                borderMode=cv2.BORDER_REFLECT)
        return rotated.astype(np.float32)
    
    def train_fast_model(self, pairs, labels):
        """Fast training with transfer learning"""
        print("\\nTraining fast Siamese network with transfer learning...")
        
        # Prepare data
        left_images = np.array([pair[0] for pair in pairs], dtype=np.float32)
        right_images = np.array([pair[1] for pair in pairs], dtype=np.float32)
        labels_array = np.array(labels, dtype=np.float32)
        
        # Simple train-test split
        indices = np.arange(len(labels_array))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=labels_array
        )
        
        X_train_left = left_images[train_idx]
        X_train_right = right_images[train_idx]
        y_train = labels_array[train_idx]
        
        X_test_left = left_images[test_idx]
        X_test_right = right_images[test_idx]
        y_test = labels_array[test_idx]
        
        print(f"Training: {len(y_train)} pairs, Testing: {len(y_test)} pairs")
        print(f"Train class distribution: {np.bincount(y_train.astype(int))}")
        print(f"Test class distribution: {np.bincount(y_test.astype(int))}")
        
        # Build and compile model
        model = self.siamese_net.build_siamese_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Higher LR for faster convergence
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        # Fast training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=5,  # Shorter patience
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,  # Shorter patience
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f"{self.models_dir}/best_fast_model.h5",
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Fast training
        history = model.fit(
            [X_train_left, X_train_right], y_train,
            batch_size=16,
            epochs=20,  # Fewer epochs
            validation_split=0.15,  # Less validation data
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_results = model.evaluate([X_test_left, X_test_right], y_test, verbose=0)
        test_loss, test_acc, test_precision, test_recall, test_auc = test_results
        
        print(f"\\nFinal Test Results:")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        
        # Detailed analysis
        test_predictions = model.predict([X_test_left, X_test_right], verbose=0)
        test_pred_binary = (test_predictions > 0.5).astype(int).flatten()
        
        # Calculate metrics
        tp = np.sum((test_pred_binary == 1) & (y_test == 1))
        tn = np.sum((test_pred_binary == 0) & (y_test == 0))
        fp = np.sum((test_pred_binary == 1) & (y_test == 0))
        fn = np.sum((test_pred_binary == 0) & (y_test == 1))
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print(f"\\nDetailed Analysis:")
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"False Acceptance Rate: {far:.4f}")
        print(f"False Rejection Rate: {frr:.4f}")
        
        # Save if decent performance (lower threshold for small datasets)
        if test_auc > 0.65 and test_acc > 0.55:
            self.siamese_net.save_model(self.models_dir)
            
            metadata = {
                'test_accuracy': float(test_acc),
                'test_auc': float(test_auc),
                'test_precision': float(test_precision),
                'test_recall': float(test_recall),
                'false_acceptance_rate': float(far),
                'false_rejection_rate': float(frr),
                'model_type': 'fast_transfer_learning',
                'training_pairs': len(y_train)
            }
            
            with open(f"{self.models_dir}/fast_model_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\\nModel saved with AUC: {test_auc:.4f}")
            return True, history, metadata
        else:
            print(f"\\nModel performance acceptable but could be better (AUC: {test_auc:.4f}, Acc: {test_acc:.4f})")
            print("Consider adding more diverse signature data.")
            return False, history, None
    
    def plot_results(self, history):
        """Quick results visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy and Loss
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history.history['auc'], label='Training AUC')
        ax2.plot(history.history['val_auc'], label='Validation AUC')
        ax2.set_title('Model AUC')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.models_dir}/fast_training_results.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def run_training(self):
        """Complete fast training pipeline"""
        print("="*70)
        print("FAST SIGNATURE VERIFICATION - TRAINING")
        print("="*70)
        
        try:
            # Load data
            user_signatures = self.load_data_optimized()
            
            if len(user_signatures) < 2:
                print("Error: Need at least 2 users for training!")
                return False
            
            # Create balanced pairs
            pairs, labels = self.create_balanced_pairs(user_signatures)
            
            if len(pairs) < 10:
                print("Error: Not enough pairs for training!")
                return False
            
            # Light augmentation
            augmented_pairs, augmented_labels = self.minimal_augmentation(pairs, labels)
            
            # Fast training
            success, history, metadata = self.train_fast_model(augmented_pairs, augmented_labels)
            
            # Plot results
            if history:
                self.plot_results(history)
            
            if success:
                print("\\n" + "="*70)
                print("FAST TRAINING COMPLETED SUCCESSFULLY!")
                print("="*70)
                print(f"Final Test AUC: {metadata['test_auc']:.4f}")
                print(f"Final Test Accuracy: {metadata['test_accuracy']:.4f}")
                print(f"False Acceptance Rate: {metadata['false_acceptance_rate']:.4f}")
                print(f"False Rejection Rate: {metadata['false_rejection_rate']:.4f}")
                print("\\nModel should now work better than the previous version!")
            else:
                print("\\nTraining completed but consider adding more users for better performance.")
            
            return True
                
        except Exception as e:
            print(f"\\nFast training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main training function"""
    trainer = FastTrainer()
    success = trainer.run_training()
    
    if not success:
        print("\\nTroubleshooting tips:")
        print("1. Ensure you have at least 2 users with 3+ signatures each")
        print("2. Check image quality and file formats")
        print("3. Verify the data/users directory structure")

if __name__ == "__main__":
    main()