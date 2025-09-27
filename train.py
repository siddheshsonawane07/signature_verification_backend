import torch
import torch.nn as nn
import os
import pandas as pd
import time
import sys
import matplotlib.pyplot as plt
import kagglehub
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from network import snn  # Siamese Neural Network architecture
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# Set device to GPU if available, otherwise CPU
# GPU acceleration significantly speeds up neural network training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download the signature verification dataset from Kaggle
# This dataset contains pairs of signature images with labels:
# - Label 0: Genuine pair (same person's signatures)
# - Label 1: Forged pair (one genuine, one forged signature)
path = kagglehub.dataset_download("robinreni/signature-verification-dataset")

class dataset(Dataset):
    """
    Custom PyTorch Dataset for signature verification pairs.
    
    Mathematical Context:
    - Each sample consists of two signature images (x1, x2) and a binary label y
    - y = 0 if signatures are from the same person (genuine pair)
    - y = 1 if one signature is forged (forged pair)
    - The goal is to learn a function f(x1, x2) → [0,1] that predicts similarity
    """
    
    def __init__(self, pairs_data, root_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            pairs_data: DataFrame with columns [img1_path, img2_path, label]
            root_dir: Root directory containing the image files
            transform: Image preprocessing transformations
        """
        self.pairs_frame = pairs_data.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """Return the total number of image pairs in the dataset."""
        return len(self.pairs_frame)

    def __getitem__(self, idx):
        """
        Retrieve a single sample (image pair + label).
        
        Mathematical Processing:
        1. Load two grayscale images: I1, I2 ∈ ℝ^(H×W)
        2. Apply transformations T: I → I' (resize, normalize, etc.)
        3. Convert to tensors for neural network processing
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            tuple: (img1_tensor, img2_tensor, label_tensor)
        """
        # Construct full file paths
        img1_name = os.path.join(self.root_dir, self.pairs_frame.iloc[idx, 0])
        img2_name = os.path.join(self.root_dir, self.pairs_frame.iloc[idx, 1])
        
        # Load images and convert to grayscale
        # Grayscale reduces dimensionality while preserving signature structure
        img1 = Image.open(img1_name).convert("L")
        img2 = Image.open(img2_name).convert("L")
        
        # Get the binary label (0 = genuine, 1 = forged)
        label = float(self.pairs_frame.iloc[idx, 2])

        # Apply preprocessing transformations if specified
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

def create_simple_split():
    """
    Create a clean train/test split from the training data only.
    
    Mathematical Rationale:
    - Stratified sampling ensures balanced class distribution in both splits
    - If original has ratio r = P(y=0)/P(y=1), both splits maintain this ratio
    - This prevents class imbalance issues during training and evaluation
    
    Returns:
        tuple: (train_dataframe, test_dataframe)
    """
    # Load only the training data to avoid problematic test split
    all_data = pd.read_csv(f'{path}/sign_data/train_data.csv', 
                          header=None, 
                          names=['img1', 'img2', 'label'])
    
    print(f"Total samples: {len(all_data)}")
    print(f"Genuine pairs (label=0): {(all_data['label'] == 0).sum()}")
    print(f"Forged pairs (label=1): {(all_data['label'] == 1).sum()}")
    
    # Create stratified split to maintain class balance
    # Mathematical: P_train(y=k) ≈ P_test(y=k) ≈ P_original(y=k) for all classes k
    train_data, test_data = train_test_split(
        all_data, 
        test_size=0.2,  # 80% train, 20% test
        stratify=all_data['label'],  # Maintain class distribution
        random_state=42  # Reproducible results
    )
    
    print(f"\nAfter clean split:")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    return train_data, test_data

# Image preprocessing pipeline
# Mathematical transformations applied to each image:
transform = transforms.Compose([
    # 1. Resize: Bilinear interpolation to standardize dimensions
    #    I_resized = Resize(I, (32, 32)) where I ∈ ℝ^(H×W) → ℝ^(32×32)
    transforms.Resize((32, 32)),  # Standardize input size for CNN
    
    # 2. Convert to tensor: Normalize pixel values from [0, 255] to [0, 1]
    #    I_tensor = I_PIL / 255.0, shape becomes (1, 32, 32) for grayscale
    transforms.ToTensor()
])

def train():
    """
    Main training loop implementing Siamese Network learning.
    
    Mathematical Framework:
    - Siamese network learns embedding function φ: ℝ^(C×H×W) → ℝ^d
    - Distance metric: D(φ(x1), φ(x2)) measures signature similarity
    - Binary classification: P(same_person | x1, x2) = σ(f(D(φ(x1), φ(x2))))
    - Loss function: BCE minimizes -Σ[y*log(p) + (1-y)*log(1-p)]
    """
    
    # Hyperparameters
    epochs = 30  # Number of complete passes through training data
    lr = 1e-4    # Learning rate: step size for gradient descent
    print(f"Learning rate: {lr}")
    batch_size = 64  # Number of samples processed simultaneously
    
    # Create clean data split
    train_df, test_df = create_simple_split()
    
    # Initialize datasets
    # Both use same root directory since we're using train folder only
    training_data = dataset(pairs_data=train_df, 
                           root_dir=f'{path}/sign_data/train/', 
                           transform=transform)
    val_data = dataset(pairs_data=test_df, 
                      root_dir=f'{path}/sign_data/train/', 
                      transform=transform)

    # Data loaders for efficient batch processing
    # Mathematical: Mini-batch gradient descent processes B samples at a time
    # This provides computational efficiency and regularization benefits
    train_loader = DataLoader(training_data, 
                             batch_size=batch_size, 
                             shuffle=True,  # Random sampling reduces overfitting
                             num_workers=4, 
                             pin_memory=True)  # Faster GPU transfer
    val_loader = DataLoader(val_data, 
                           batch_size=64, 
                           shuffle=False,  # No need to shuffle validation
                           num_workers=4, 
                           pin_memory=True)
    
    # Initialize Siamese Neural Network
    model = snn().to(device)
    
    # Adam optimizer: Adaptive learning rate with momentum
    # Mathematical: Uses first and second moment estimates of gradients
    # θ_{t+1} = θ_t - α * m̂_t / (√v̂_t + ε)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Binary Cross-Entropy Loss with Logits
    # Mathematical: L = -Σ[y*log(σ(z)) + (1-y)*log(1-σ(z))]
    # where z is the raw logit output, σ is sigmoid function
    # Numerically stable version that combines sigmoid and BCE
    criterion = nn.BCEWithLogitsLoss()
    
    start_time = time.time()
    best_accuracy = 0
    
    # Lists to track training progress
    loss_list = []
    acc_list = []

    # Main training loop
    for epoch in range(epochs):
        print('-'*90)
        print(f"Epoch {epoch+1}:")
        
        # Set model to training mode
        # Enables dropout, batch normalization updates, etc.
        model.train()
        total_loss = 0
        
        # Process training batches
        for i, (img1, img2, label) in tqdm(enumerate(train_loader), 
                                          total=len(train_loader), 
                                          desc='Training'):
            # Move data to GPU if available
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            # Forward pass: compute model predictions
            # Mathematical: output = f(φ(img1), φ(img2))
            # where φ is the feature extraction network
            output = model(img1, img2)
            
            # Compute loss between predictions and ground truth
            # Mathematical: L = BCE(output, label)
            loss = criterion(output.squeeze(), label)
            
            # Backward pass: compute gradients
            # Mathematical: ∇θ L = ∂L/∂θ via automatic differentiation
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()        # Compute gradients
            
            # Update parameters using Adam optimizer
            # Mathematical: θ = θ - lr * update_rule(∇θ L)
            optimizer.step()

            total_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # Validate model performance on test set
        acc = validate(model, criterion, val_loader)
        
        # Save best model based on validation accuracy
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), 'best_model.pth')
            
        # Track metrics for plotting
        loss_list.append(avg_loss)
        acc_list.append(acc)
        
    end_time = time.time()
    
    # Save final model
    torch.save(model.state_dict(), 'model_last.pth')
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    print(f"Best accuracy achieved: {best_accuracy:.4f}")
    
    # Plot training progress
    plot_metrics(loss_list, acc_list)

def validate(model, criterion, val_loader):
    """
    Evaluate model performance on validation set.
    
    Mathematical Evaluation:
    - Accuracy = (TP + TN) / (TP + TN + FP + FN)
    - Where TP=True Positive, TN=True Negative, etc.
    - For signature verification:
      * TP: Correctly identified genuine pairs
      * TN: Correctly identified forged pairs
      * FP: Genuine pairs classified as forged
      * FN: Forged pairs classified as genuine
    
    Args:
        model: Trained neural network
        criterion: Loss function
        val_loader: Validation data loader
        
    Returns:
        float: Validation accuracy
    """
    # Set model to evaluation mode
    # Disables dropout, uses population statistics for batch norm, etc.
    model.eval()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for img1, img2, label in tqdm(val_loader, desc='Validation'):
            # Move data to device
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            # Forward pass: get model predictions
            output = model(img1, img2)
            
            # Compute validation loss
            loss = criterion(output.squeeze(), label)
            total_loss += loss.item()

            # Convert logits to probabilities and make binary predictions
            # Mathematical: 
            # 1. p = σ(logit) = 1/(1 + e^(-logit)) ∈ [0,1]
            # 2. prediction = round(p) = {0 if p < 0.5, 1 if p ≥ 0.5}
            pred = torch.sigmoid(output).round()
            
            # Count correct predictions
            total_correct += pred.eq(label.view_as(pred)).sum().item()
            total_samples += label.size(0)

        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
    
    return accuracy

def test_all():
    """
    Evaluate the trained model on the test set.
    
    This function loads the final model and evaluates its performance
    on the test split created during data preprocessing.
    """
    # Use the same data split for consistency
    _, test_df = create_simple_split()
    val_data = dataset(pairs_data=test_df, 
                      root_dir=f'{path}/sign_data/train/', 
                      transform=transform)
    val_loader = DataLoader(val_data, 
                           batch_size=64, 
                           shuffle=False, 
                           num_workers=4, 
                           pin_memory=True)
    
    # Load trained model
    model = snn().to(device)
    model.load_state_dict(torch.load('model_last.pth', weights_only=True))
    criterion = nn.BCEWithLogitsLoss()
    
    # Evaluate performance
    acc = validate(model, criterion, val_loader)
    print(f"Final Test Accuracy: {acc:.4f}")

def image_similarity(img1_path, img2_path):
    """
    Predict similarity between two signature images.
    
    Mathematical Process:
    1. Load and preprocess both images: I1, I2 → T(I1), T(I2)
    2. Extract features: φ(T(I1)), φ(T(I2))
    3. Compute similarity: s = σ(f(φ(T(I1)), φ(T(I2))))
    4. Interpret: s < 0.5 → genuine pair, s ≥ 0.5 → forged pair
    
    Args:
        img1_path: Path to first signature image
        img2_path: Path to second signature image
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Load trained model
    model = snn().to(device)
    model.load_state_dict(torch.load('model_last.pth', weights_only=True))
    model.eval()  # Set to evaluation mode

    # Preprocess images
    # Mathematical: Apply same transformations as training data
    img1 = transform(Image.open(img1_path).convert("L")).unsqueeze(0).to(device)
    img2 = transform(Image.open(img2_path).convert("L")).unsqueeze(0).to(device)

    # Get prediction without gradient computation
    with torch.no_grad():
        # Forward pass through Siamese network
        logit = model(img1, img2)
        
        # Convert logit to probability
        # Mathematical: probability = σ(logit) = 1/(1 + e^(-logit))
        similarity = torch.sigmoid(logit).item()

    return similarity

def plot_metrics(loss, acc):
    """
    Visualize training progress with loss and accuracy curves.
    
    Mathematical Interpretation:
    - Training loss should generally decrease over epochs (convergence)
    - Validation accuracy should generally increase (learning)
    - Large gaps might indicate overfitting
    - Plateaus might indicate convergence or need for hyperparameter tuning
    """
    epochs = range(1, len(loss) + 1)

    # Create subplot with dual y-axes for loss and accuracy
    fig, ax1 = plt.subplots()

    # Plot training loss on left y-axis
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, loss, label='Training Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot validation accuracy on right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:orange')
    ax2.plot(epochs, acc, label='Validation Accuracy', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Format and save plot
    fig.tight_layout()
    plt.title('Training Loss and Validation Accuracy over Epochs')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.savefig('metrics.png')
    plt.show()

if __name__ == "__main__":
    """
    Command-line interface for different operations:
    - train: Train the Siamese network from scratch
    - val: Evaluate trained model on test set  
    - test img1 img2: Compare two signature images
    """
    if len(sys.argv) < 2:
        print("Usage: python train.py [train | val | test img1 img2]")
        sys.exit(1)
        
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'val':
        test_all()
    elif sys.argv[1] == 'test' and len(sys.argv) == 4:
        similarity = image_similarity(sys.argv[2], sys.argv[3])
        # Decision boundary at 0.5: below = genuine, above = forged
        print('Genuine' if similarity < 0.5 else 'Forged')
    else:
        print("Usage: python train.py [train | val | test img1 img2]")

"""
MATHEMATICAL SUMMARY:

1. SIAMESE NETWORK ARCHITECTURE:
   - Two identical CNNs share weights: φ_θ(x1) and φ_θ(x2)
   - Feature vectors are combined: h = combine(φ_θ(x1), φ_θ(x2))
   - Final prediction: p = σ(W·h + b) where σ is sigmoid

2. LOSS FUNCTION:
   - Binary Cross-Entropy: L = -Σ[y*log(p) + (1-y)*log(1-p)]
   - Minimizes classification error between genuine/forged pairs

3. OPTIMIZATION:
   - Adam optimizer: Adaptive learning rate with momentum
   - Mini-batch gradient descent for computational efficiency

4. EVALUATION METRICS:
   - Accuracy: (Correct Predictions) / (Total Predictions)
   - Decision boundary at p = 0.5 for binary classification

5. DATA PREPROCESSING:
   - Resize to 32x32 pixels for computational efficiency
   - Normalize pixel values to [0,1] for stable training
   - Stratified split maintains class balance
"""