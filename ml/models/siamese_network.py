"""
Fast Siamese Neural Network for Signature Verification
Uses transfer learning for rapid training with good performance
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class SiameseNetwork:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        
    def create_base_network(self):
        """Create lightweight base network with transfer learning"""
        # Use pre-trained MobileNetV2 - fast and effective
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model to speed up training
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            # Resize input to match MobileNet requirements
            layers.Resizing(224, 224),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu')
        ])
        return model
    
    def build_siamese_model(self):
        """Build Siamese network with two inputs"""
        
        # Create base network
        base_network = self.create_base_network()
        
        # Two input branches
        input_a = layers.Input(shape=self.input_shape)
        input_b = layers.Input(shape=self.input_shape)
        
        # Process both inputs through same network
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        
        # Calculate distance between features
        distance = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([processed_a, processed_b])
        
        # Output layer for similarity
        output = layers.Dense(1, activation='sigmoid')(distance)
        
        # Create model
        self.model = Model(inputs=[input_a, input_b], outputs=output)
        
        # Don't compile here - let training script handle it
        return self.model
    
    def get_feature_extractor(self):
        """Get just the feature extraction part"""
        if self.model is None:
            self.build_siamese_model()
        
        base_network = self.create_base_network()
        return base_network
    
    def save_model(self, models_dir):
        """Save the trained model"""
        if self.model is not None:
            model_path = f"{models_dir}/fast_siamese_model.h5"
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save. Train the model first.")
    
    def load_model(self, model_path):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(model_path)
        return self.model