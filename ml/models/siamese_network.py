"""
Simplified Siamese Neural Network for Signature Verification
Uses transfer learning for better performance with limited data
Compatible with existing SignatureVerifier and training pipeline
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime

class SiameseNetwork:
    def __init__(self, input_shape=(224, 224, 3), models_dir="ml/training/data/models"):
        self.input_shape = input_shape
        self.models_dir = Path(models_dir)
        self.model = None
        self.base_network = None
        
        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def create_base_network(self):
        """Create simplified base network with transfer learning"""
        # Use pre-trained MobileNetV2 - proven to work well with limited data
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze most layers to prevent overfitting
        base_model.trainable = False
        
        # Simple feature extraction head
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu', name='features_128'),
            layers.Dropout(0.1),
            layers.Dense(64, activation='relu', name='features_64')
        ], name='signature_backbone')
        
        self.base_network = model
        return model
    
    def build_siamese_model(self):
        """Build simplified Siamese network"""
        
        # Create base network
        base_network = self.create_base_network()
        
        # Two input branches
        input_a = layers.Input(shape=self.input_shape, name='signature_a')
        input_b = layers.Input(shape=self.input_shape, name='signature_b')
        
        # Process both inputs through same network
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        
        # Simple distance calculation
        distance = layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]), 
            name='l1_distance'
        )([processed_a, processed_b])
        
        # Simple decision network
        x = layers.Dense(32, activation='relu', name='decision_32')(distance)
        x = layers.Dropout(0.3, name='decision_dropout')(x)
        x = layers.Dense(16, activation='relu', name='decision_16')(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='similarity_output')(x)
        
        # Create model
        self.model = Model(
            inputs=[input_a, input_b], 
            outputs=output, 
            name='siamese_signature_model'
        )
        
        return self.model
    
    def get_feature_extractor(self):
        """Get just the feature extraction part"""
        if self.base_network is None:
            self.create_base_network()
        return self.base_network
    
    def save_model(self, models_dir=None, model_name="siamese_signature_model"):
        """Save the trained model to match training pipeline format"""
        if models_dir is None:
            models_dir = self.models_dir
        else:
            models_dir = Path(models_dir)
            
        if self.model is not None:
            model_path = models_dir / f"{model_name}.h5"
            backbone_path = models_dir / f"signature_backbone.h5"
            
            # Save main model
            self.model.save(str(model_path))
            print(f"Model saved to {model_path}")
            
            # Save backbone separately
            if self.base_network is not None:
                self.base_network.save(str(backbone_path))
                print(f"Backbone saved to {backbone_path}")
                
            return str(model_path)
        else:
            print("No model to save. Train the model first.")
            return None
    
    def load_model(self, model_path):
        """Load a saved model"""
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            print(f"Model loaded from {model_path}")
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None