"""
Siamese Neural Network for Signature Verification
Enhanced architecture with EfficientNet backbone and attention mechanisms
Environment versions matching specified environment.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.applications import EfficientNetB3
import numpy as np


class SiameseNetwork:
    def __init__(self, input_shape=(224, 224, 3)):
        """
        Initialize Siamese Network with input shape.
        Args:
            input_shape (tuple): Shape of input images (height, width, channels).
        """
        self.input_shape = input_shape
        self.feature_extractor = None
        self.siamese_model = None
        self.model_built = False

    def build_feature_extractor(self):
        """
        Build feature extractor using EfficientNetB3 pretrained on ImageNet.
        Adds a custom head with attention mechanisms and L2 normalization.
        
        Returns:
            keras Model: Feature extractor model that outputs 256-dimensional normalized features.
        """
        # Load EfficientNetB3 base model without top classification layer
        # Uses ImageNet weights for transfer learning
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,  # Remove final classification layer
            input_shape=self.input_shape
        )

        # Initially freeze the EfficientNetB3 weights to prevent overwriting pretrained features
        base_model.trainable = False

        # Add global average pooling to reduce spatial dimensions from (H, W, C) to (C,)
        x = base_model.output
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)

        # Implement attention mechanism to focus on important features
        attention_weights = layers.Dense(
            x.shape[-1],  # Same number of features as input
            activation='softmax',  # Ensures attention weights sum to 1
            name='attention_weights'
        )(x)

        # Apply attention weights element-wise to original features
        attended_features = layers.Multiply(name='attention_features')([x, attention_weights])

        # Concatenate original and attended features for richer representation
        x = layers.Concatenate(name='feature_concat')([x, attended_features])

        # Add regularization and feature processing layers
        x = layers.Dropout(0.4, name='dropout_1')(x)  # Prevent overfitting
        x = layers.Dense(1024, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization(name='batch_norm_1')(x)  # Stabilize training
        x = layers.Dropout(0.3, name='dropout_2')(x)
        x = layers.Dense(512, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization(name='batch_norm_2')(x)
        x = layers.Dropout(0.2, name='dropout_3')(x)

        # Final feature embedding of size 256
        features = layers.Dense(256, activation='relu', name='feature_output')(x)

        # L2 normalize features to unit length for better similarity computation
        # This ensures cosine similarity and euclidean distance are related
        normalized_features = layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1),
            name='l2_normalize'
        )(features)

        # Build the complete feature extractor model
        self.feature_extractor = Model(
            inputs=base_model.input,
            outputs=normalized_features,
            name='signature_feature_extractor'
        )

        return self.feature_extractor

    def build_siamese_model(self):
        """
        Build complete Siamese network architecture for signature pair comparison.
        Uses shared feature extractor and multiple similarity metrics for robust comparison.

        Returns:
            keras Model: Complete Siamese network model with binary similarity output.
        """
        if self.feature_extractor is None:
            self.build_feature_extractor()

        # Define two input branches for signature pairs
        input_a = layers.Input(shape=self.input_shape, name='signature_a')
        input_b = layers.Input(shape=self.input_shape, name='signature_b')

        # Extract features from both signatures using shared feature extractor
        # Weight sharing ensures consistent feature extraction
        features_a = self.feature_extractor(input_a)
        features_b = self.feature_extractor(input_b)

        # Compute multiple similarity metrics for robust comparison

        # 1. Euclidean distance: L2 distance between feature vectors
        euclidean_distance = layers.Lambda(
            lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True)),
            name='euclidean_distance'
        )([features_a, features_b])

        # 2. Cosine similarity: Dot product of normalized vectors
        # Since features are L2-normalized, this gives cosine similarity directly
        cosine_similarity = layers.Lambda(
            lambda x: tf.reduce_sum(x[0] * x[1], axis=1, keepdims=True),
            name='cosine_similarity'
        )([features_a, features_b])

        # 3. Element-wise absolute difference: Captures feature-wise differences
        abs_diff = layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]),
            name='absolute_difference'
        )([features_a, features_b])

        # 4. Element-wise product: Captures feature interactions
        element_product = layers.Lambda(
            lambda x: x[0] * x[1],
            name='element_product'
        )([features_a, features_b])

        # Combine all similarity metrics into a single feature vector
        combined_features = layers.Concatenate(name='combined_similarity')([
            euclidean_distance,
            cosine_similarity,
            abs_diff,
            element_product
        ])

        # Decision network to learn optimal combination of similarity metrics
        x = layers.Dense(512, activation='relu', name='decision_dense_1')(combined_features)
        x = layers.Dropout(0.3, name='decision_dropout_1')(x)
        x = layers.Dense(256, activation='relu', name='decision_dense_2')(x)
        x = layers.Dropout(0.2, name='decision_dropout_2')(x)
        x = layers.Dense(128, activation='relu', name='decision_dense_3')(x)

        # Final similarity score between 0 and 1
        # 1 indicates genuine pair, 0 indicates forged pair
        similarity_output = layers.Dense(
            1,
            activation='sigmoid',
            name='similarity_score'
        )(x)

        # Build complete Siamese model
        self.siamese_model = Model(
            inputs=[input_a, input_b],
            outputs=similarity_output,
            name='siamese_signature_verifier'
        )

        self.model_built = True
        return self.siamese_model

    def compile_model(self, learning_rate=0.001):
        """
        Compile Siamese model with optimizer, loss function, and evaluation metrics.

        Args:
            learning_rate (float): Learning rate for Adam optimizer.

        Returns:
            keras Model: Compiled Siamese model ready for training.
        """
        if not self.model_built:
            self.build_siamese_model()

        # Compile model with binary crossentropy loss for similarity classification
        self.siamese_model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',  # Suitable for binary similarity classification
            metrics=[
                'accuracy',  # Overall accuracy
                tf.keras.metrics.Precision(name='precision'),  # Precision for genuine pairs
                tf.keras.metrics.Recall(name='recall'),        # Recall for genuine pairs
                tf.keras.metrics.AUC(name='auc')              # Area under ROC curve
            ]
        )

        return self.siamese_model

    def create_training_pairs(self, signature_paths, labels):
        """
        Create positive (genuine) and negative (forged) pairs for Siamese network training.
        
        Args:
            signature_paths (list): List of paths to signature images.
            labels (list): Corresponding user labels for each signature.

        Returns:
            tuple: (pairs, pair_labels) where pairs is list of [path1, path2] 
                   and pair_labels is list of 1 (genuine) or 0 (forged).
        """
        pairs = []
        pair_labels = []

        # Group signatures by user to create positive and negative pairs
        user_signatures = {}
        for path, label in zip(signature_paths, labels):
            if label not in user_signatures:
                user_signatures[label] = []
            user_signatures[label].append(path)

        # Create positive pairs (same user signatures)
        for user, sigs in user_signatures.items():
            # Generate all combinations of signatures for this user
            for i in range(len(sigs)):
                for j in range(i + 1, len(sigs)):
                    pairs.append([sigs[i], sigs[j]])
                    pair_labels.append(1)  # Label 1 for genuine pairs

        # Create negative pairs (different user signatures)
        users = list(user_signatures.keys())
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user1_sigs = user_signatures[users[i]]
                user2_sigs = user_signatures[users[j]]

                # Limit negative samples to prevent class imbalance
                # Take only first 2 signatures from each user for negative pairs
                for sig1 in user1_sigs[:2]:
                    for sig2 in user2_sigs[:2]:
                        pairs.append([sig1, sig2])
                        pair_labels.append(0)  # Label 0 for forged pairs

        return pairs, pair_labels

    def fine_tune_feature_extractor(self, epochs=10, learning_rate=0.0001):
        """
        Enable fine-tuning of EfficientNetB3 backbone by unfreezing top layers.
        This allows the model to adapt pretrained features to signature verification task.

        Args:
            epochs (int): Number of epochs for fine-tuning phase.
            learning_rate (float): Lower learning rate for stable fine-tuning.
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not built yet")

        # Access the EfficientNetB3 base model within feature extractor
        base_model = self.feature_extractor.layers[1]  # EfficientNetB3 is layer 1

        # Unfreeze the entire base model
        base_model.trainable = True

        # Keep early layers frozen, only fine-tune top 20 layers
        # This preserves low-level features while adapting high-level features
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        # Recompile with lower learning rate for stable fine-tuning
        self.siamese_model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )

        trainable_layers = len([l for l in base_model.layers if l.trainable])
        print(f"Fine-tuning enabled for {trainable_layers} layers")

    def save_model(self, model_path="../data/models/"):
        """
        Save trained models and metadata to disk for future use.

        Args:
            model_path (str): Directory path to save models and metadata.
        """
        import os
        os.makedirs(model_path, exist_ok=True)

        # Save complete Siamese model
        self.siamese_model.save(f"{model_path}/siamese_model.h5")

        # Save feature extractor separately for standalone feature extraction
        self.feature_extractor.save(f"{model_path}/feature_extractor.h5")

        # Save model metadata for version tracking and compatibility
        metadata = {
            'input_shape': self.input_shape,
            'feature_size': 256,
            'model_version': '1.0',
            'tensorflow_version': tf.__version__,
            'creation_date': float(tf.timestamp().numpy())
        }

        import json
        with open(f"{model_path}/model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Models saved to {model_path}")

    def load_model(self, model_path="../data/models/"):
        """
        Load pre-trained models from disk.

        Args:
            model_path (str): Directory path containing saved models.
        """
        try:
            # Load both models from saved files
            self.siamese_model = keras.models.load_model(f"{model_path}/siamese_model.h5")
            self.feature_extractor = keras.models.load_model(f"{model_path}/feature_extractor.h5")
            self.model_built = True
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Building new models...")
            # If loading fails, build and compile new models
            self.build_siamese_model()
            self.compile_model()

    def get_model_summary(self):
        """
        Display detailed architecture information for both models.
        Includes layer details and parameter counts.
        """
        if self.siamese_model is None:
            return "Model not built yet"

        print("=== SIAMESE MODEL ARCHITECTURE ===")
        self.siamese_model.summary()

        print("\n=== FEATURE EXTRACTOR ARCHITECTURE ===")
        self.feature_extractor.summary()

        # Calculate and display parameter statistics
        total_params = self.siamese_model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.siamese_model.trainable_weights])

        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize Siamese network for signature verification
    siamese_net = SiameseNetwork()
    
    # Build the complete architecture
    model = siamese_net.build_siamese_model()
    
    # Compile with optimizer and metrics
    compiled_model = siamese_net.compile_model()
    
    # Display architecture information
    siamese_net.get_model_summary()
    
    print("Siamese network ready for training!")
