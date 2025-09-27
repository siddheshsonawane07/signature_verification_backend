import torch
import torch.nn as nn
import torch.nn.functional as F

class snn(nn.Module):
    """
    Siamese Neural Network for signature verification.
    
    Mathematical Framework:
    A Siamese network consists of two identical subnetworks (twins) that share weights.
    For signature verification:
    
    1. Feature Extraction: φ_θ: ℝ^(C×H×W) → ℝ^d
       Both signatures are processed by the same CNN to extract feature vectors
    
    2. Similarity Computation: D: ℝ^d × ℝ^d → ℝ^d
       The L1 distance (absolute difference) measures feature dissimilarity
    
    3. Binary Classification: f: ℝ^d → ℝ
       A fully connected layer maps the distance to a similarity score
    
    Architecture: Input → CNN → Feature Vector → L1 Distance → FC → Logit
    """
    
    def __init__(self) -> None:
        """
        Initialize the Siamese Neural Network.
        
        Architecture Components:
        - cnn: Shared feature extractor (identical weights for both inputs)
        - fc1: Final classification layer mapping distance vector to similarity score
        """
        super(snn, self).__init__()
        
        # Shared CNN feature extractor
        # Mathematical: φ_θ where θ represents learnable parameters
        # Same network processes both signature images
        self.cnn = cnn()
        
        # Final classification layer
        # Mathematical: f(z) = W·z + b where z is the L1 distance vector
        # Maps 128-dimensional distance vector to single similarity score
        self.fc1 = nn.Linear(128, 1)
        
    def forward(self, x, y):
        """
        Forward pass through the Siamese network.
        
        Mathematical Process:
        1. Feature extraction: v1 = φ_θ(x), v2 = φ_θ(y)
        2. Distance computation: z = |v1 - v2| (element-wise L1 distance)
        3. Similarity prediction: s = f(z) = W·z + b
        
        Args:
            x: First signature image tensor, shape (batch_size, 1, 32, 32)
            y: Second signature image tensor, shape (batch_size, 1, 32, 32)
            
        Returns:
            torch.Tensor: Similarity logits, shape (batch_size, 1)
                         Values close to 0 indicate genuine pairs
                         Values close to 1 indicate forged pairs
        """
        # Extract features from both signature images using shared CNN
        # Mathematical: v1 = φ_θ(x) ∈ ℝ^128
        result1 = self.cnn(x)
        
        # Mathematical: v2 = φ_θ(y) ∈ ℝ^128  
        # Note: Same network φ_θ is used (weight sharing)
        result2 = self.cnn(y)
        
        # Compute L1 (Manhattan) distance between feature vectors
        # Mathematical: z = |v1 - v2| = (|v1[1] - v2[1]|, ..., |v1[128] - v2[128]|)
        # 
        # Why L1 distance?
        # - Captures feature-wise dissimilarity between signatures
        # - More robust to outliers than L2 (Euclidean) distance
        # - Each dimension contributes equally to the final distance
        # - Provides sparse gradients which can improve training
        z = torch.abs(result1 - result2)  # Element-wise absolute difference
        
        # Map distance vector to similarity score
        # Mathematical: similarity_logit = W·z + b
        # where W ∈ ℝ^(1×128), b ∈ ℝ, z ∈ ℝ^128
        z = self.fc1(z)
        
        return z


class cnn(nn.Module):
    """
    Convolutional Neural Network for feature extraction from signature images.
    
    Mathematical Architecture:
    Input: I ∈ ℝ^(1×32×32) (grayscale signature image)
    
    Layer-by-layer transformations:
    1. Conv1 + ReLU + Pool1: (1,32,32) → (32,16,16)
    2. Conv2 + ReLU + Pool2: (32,16,16) → (64,8,8)  
    3. Conv3 + ReLU + Pool3: (64,8,8) → (128,4,4)
    4. Flatten + FC1: (128,4,4) → (2048,) → (128,)
    
    Output: Feature vector φ(I) ∈ ℝ^128
    """
    
    def __init__(self) -> None:
        """
        Initialize the CNN feature extractor.
        
        Design Principles:
        - Progressive feature extraction: edges → textures → patterns
        - Spatial downsampling: Reduces computational cost while preserving information
        - Channel expansion: Increases representational capacity at deeper layers
        """
        super(cnn, self).__init__()
        
        # First Convolutional Block
        # Mathematical: Conv1D(I) = ReLU(I * W1 + b1)
        # Input: 1 channel (grayscale), Output: 32 feature maps
        # Kernel: 3×3 with padding=1 preserves spatial dimensions
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, 
                              kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Spatial downsampling by 2
        
        # Second Convolutional Block  
        # Mathematical: Conv2(x) = ReLU(x * W2 + b2)
        # Extracts more complex features from the 32 feature maps
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, 
                              kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Further downsampling
        
        # Third Convolutional Block
        # Mathematical: Conv3(x) = ReLU(x * W3 + b3)  
        # Captures high-level signature patterns and structures
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, 
                              kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Final spatial reduction
        
        # Fully Connected Layer for feature compression
        # Mathematical calculation of input size:
        # - Initial: 32×32 pixels
        # - After pool1: 32×32 → 16×16 (÷2)
        # - After pool2: 16×16 → 8×8 (÷2)  
        # - After pool3: 8×8 → 4×4 (÷2)
        # - Total features: 128 channels × 4 × 4 = 2048
        # 
        # Mathematical: FC1(x) = W·x + b where x ∈ ℝ^2048, output ∈ ℝ^128
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=128)
        
    def forward(self, x):
        """
        Forward pass through the CNN feature extractor.
        
        Mathematical Flow:
        x ∈ ℝ^(B×1×32×32) → ... → φ(x) ∈ ℝ^(B×128)
        where B is the batch size
        
        Args:
            x: Input signature image tensor, shape (batch_size, 1, 32, 32)
            
        Returns:
            torch.Tensor: Feature vector, shape (batch_size, 128)
        """
        
        # First Convolutional Block
        # Mathematical: h1 = MaxPool(ReLU(Conv1(x)))
        # Shape: (B, 1, 32, 32) → (B, 32, 32, 32) → (B, 32, 16, 16)
        x = self.conv1(x)        # Convolution: learns edge detectors
        x = F.relu(x)            # ReLU activation: f(z) = max(0, z)
        x = self.pool1(x)        # Max pooling: spatial downsampling + translation invariance
        
        # Second Convolutional Block  
        # Mathematical: h2 = MaxPool(ReLU(Conv2(h1)))
        # Shape: (B, 32, 16, 16) → (B, 64, 16, 16) → (B, 64, 8, 8)
        x = self.conv2(x)        # Convolution: learns texture patterns
        x = F.relu(x)            # Non-linear activation
        x = self.pool2(x)        # Spatial reduction
        
        # Third Convolutional Block
        # Mathematical: h3 = MaxPool(ReLU(Conv3(h2)))  
        # Shape: (B, 64, 8, 8) → (B, 128, 8, 8) → (B, 128, 4, 4)
        x = self.conv3(x)        # Convolution: learns high-level patterns
        x = F.relu(x)            # Non-linear activation  
        x = self.pool3(x)        # Final spatial reduction
        
        # Flatten spatial dimensions for fully connected layer
        # Mathematical: Reshape h3 from (B, 128, 4, 4) to (B, 2048)
        # This converts 2D feature maps to 1D feature vectors
        x = x.view(x.size(0), -1)  # Flatten: (B, 128, 4, 4) → (B, 2048)
        
        # Final feature extraction
        # Mathematical: φ(input) = FC1(flatten(h3)) ∈ ℝ^128
        # Compresses 2048-dimensional representation to 128-dimensional feature vector
        x = self.fc1(x)          # Dense layer: learns feature combinations
        
        return x

"""
MATHEMATICAL SUMMARY OF THE SIAMESE ARCHITECTURE:

1. FEATURE EXTRACTION (CNN):
   - Convolution: h_l = σ(W_l * h_{l-1} + b_l)
   - Max Pooling: Spatial downsampling for translation invariance
   - Progressive abstraction: edges → textures → shapes → signatures

2. SIAMESE COMPUTATION:
   - Weight Sharing: φ_θ(x1) and φ_θ(x2) use identical parameters θ
   - L1 Distance: z = |φ_θ(x1) - φ_θ(x2)| measures feature dissimilarity
   - Classification: y = W·z + b maps distance to similarity score

3. TRAINING OBJECTIVE:
   - Minimize: L = BCE(σ(y), label) where σ is sigmoid
   - Goal: Learn φ_θ such that similar signatures have small ||φ_θ(x1) - φ_θ(x2)||_1
   - And dissimilar signatures have large ||φ_θ(x1) - φ_θ(x2)||_1

4. SPATIAL DIMENSION FLOW:
   - Input: 32×32 → Conv+Pool → 16×16 → Conv+Pool → 8×8 → Conv+Pool → 4×4
   - Channel Expansion: 1 → 32 → 64 → 128 (increases representational capacity)
   - Final Features: 128-dimensional signature embedding

5. DESIGN ADVANTAGES:
   - Weight sharing reduces parameters and enforces symmetric similarity function
   - L1 distance is robust and provides interpretable feature differences  
   - Progressive downsampling captures multi-scale signature characteristics
   - End-to-end learning optimizes features specifically for signature verification
"""