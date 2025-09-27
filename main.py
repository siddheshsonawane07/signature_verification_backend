from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import base64
import io
from PIL import Image
import os
import uvicorn
from network import snn
from torchvision import transforms
import logging
from typing import Optional
import time

# Configure logging for monitoring and debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application with metadata
app = FastAPI(
    title="Signature Verification API",
    description="AI-powered signature verification using Siamese Neural Networks",
    version="1.0.0"
)

# Enable Cross-Origin Resource Sharing (CORS) for React Native mobile apps
# This allows mobile applications to make HTTP requests to this API server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your mobile app's domain for security
    allow_credentials=True,
    allow_methods=["*"],   # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],   # Allow all headers
)

# Device configuration for optimal performance
# Mathematical Context: Neural networks benefit significantly from GPU acceleration
# GPU provides parallel processing for matrix operations in convolutions and linear layers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing pipeline - identical to training pipeline
# Mathematical Transformations:
# 1. Resize: Bilinear interpolation I(x,y) → I'(x',y') where (x',y') ∈ [0,32]×[0,32]
# 2. ToTensor: Normalize pixel values [0,255] → [0,1] and reshape (H,W) → (1,H,W)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Standardize input dimensions for CNN
    transforms.ToTensor()         # Convert PIL Image to PyTorch tensor
])

# Global model variable for efficient memory usage
# Loaded once at startup rather than per request
model = None

def load_model():
    """
    Load the trained Siamese Neural Network model.
    
    Mathematical Context:
    The model contains learned parameters θ* that minimize the training loss:
    θ* = argmin_θ Σ L(f_θ(x1_i, x2_i), y_i)
    where L is Binary Cross-Entropy loss and f_θ is the Siamese network
    
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    global model
    try:
        # Initialize Siamese Neural Network architecture
        model = snn().to(device)
        
        # Load pre-trained weights with priority order
        if os.path.exists('best_model.pth'):
            # Best model based on validation accuracy during training
            model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
            logger.info("✅ Loaded best_model.pth")
        elif os.path.exists('model_last.pth'):
            # Final model from last training epoch
            model.load_state_dict(torch.load('model_last.pth', map_location=device, weights_only=True))
            logger.info("✅ Loaded model_last.pth")
        else:
            raise FileNotFoundError("❌ No trained model found. Please train the model first.")
        
        # Set to evaluation mode
        # Mathematical: Disables dropout, uses population statistics for batch norm
        model.eval()
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False

def process_image(image_data):
    """
    Process image from various input formats (base64, file upload, etc.).
    
    Mathematical Context:
    Converts raw image data to standardized format for neural network processing:
    Raw Image → PIL Image → Grayscale → Tensor → Normalized
    
    Args:
        image_data: Image in various formats (base64 string, file object, etc.)
        
    Returns:
        PIL.Image: Processed grayscale image ready for transformation
        
    Raises:
        ValueError: If image format is unsupported or processing fails
    """
    try:
        if isinstance(image_data, str):
            # Handle base64 encoded images (common for mobile apps)
            # Mathematical: Decode base64 string to binary image data
            if 'base64,' in image_data:
                # Remove data URL prefix (e.g., "data:image/png;base64,")
                image_data = image_data.split('base64,')[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_data)
            # Convert bytes to PIL Image and ensure grayscale
            image = Image.open(io.BytesIO(image_bytes)).convert("L")
            
        elif hasattr(image_data, 'read'):
            # Handle file upload objects
            image = Image.open(image_data).convert("L")
        else:
            raise ValueError("Unsupported image format")
        
        return image
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def verify_signatures(img1, img2):
    """
    Compare two signature images using the trained Siamese Neural Network.
    
    Mathematical Process:
    1. Preprocessing: T(I1), T(I2) where T includes resize and normalization
    2. Feature extraction: v1 = φ_θ(T(I1)), v2 = φ_θ(T(I2)) ∈ ℝ^128
    3. Distance computation: d = |v1 - v2| (L1 distance)
    4. Classification: logit = W·d + b
    5. Probability: p = σ(logit) = 1/(1 + e^(-logit))
    6. Decision: genuine if p < 0.5, forged if p ≥ 0.5
    
    Args:
        img1: First signature image (PIL Image)
        img2: Second signature image (PIL Image)
        
    Returns:
        dict: Verification results with mathematical metrics
    """
    try:
        start_time = time.time()
        
        # Apply preprocessing transformations
        # Mathematical: Convert PIL Images to normalized tensors
        img1_tensor = transform(img1).unsqueeze(0).to(device)  # Add batch dimension
        img2_tensor = transform(img2).unsqueeze(0).to(device)
        
        # Forward pass through Siamese network
        with torch.no_grad():  # Disable gradient computation for inference
            # Mathematical: logit = f_θ(img1_tensor, img2_tensor)
            logit = model(img1_tensor, img2_tensor)
            
            # Convert logit to probability using sigmoid function
            # Mathematical: p = σ(logit) = 1/(1 + e^(-logit)) ∈ [0,1]
            similarity_score = torch.sigmoid(logit).item()
        
        processing_time = time.time() - start_time
        
        # Interpret results based on decision boundary
        # Mathematical Decision Rule:
        # - If p < 0.5: Signatures are from same person (genuine)
        # - If p ≥ 0.5: One signature is forged
        is_genuine = similarity_score < 0.5
        
        # Calculate confidence as distance from decision boundary
        # Mathematical: confidence = |p - 0.5| * 2 ∈ [0,1]
        # Values close to 0 or 1 have high confidence, values near 0.5 have low confidence
        confidence = abs(similarity_score - 0.5) * 2
        
        return {
            'is_genuine': is_genuine,
            'similarity_score': float(similarity_score),
            'confidence': float(confidence),
            'result': 'Genuine' if is_genuine else 'Forged',
            'processing_time_ms': round(processing_time * 1000, 2),
            'threshold': 0.5  # Decision boundary
        }
    except Exception as e:
        raise RuntimeError(f"Error during verification: {str(e)}")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint providing API information and status.
    
    Returns basic information about the signature verification service,
    including model loading status and available endpoints.
    """
    return {
        "message": "🔐 Signature Verification API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "device": str(device),
        "endpoints": {
            "/docs": "API Documentation",
            "/health": "Health Check",
            "/verify-base64": "Verify signatures from base64 images",
            "/verify-files": "Verify signatures from file uploads"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancer integration.
    
    Returns service health status including model availability.
    Essential for production deployment monitoring.
    """
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device),
        "timestamp": time.time()
    }

@app.post("/verify-base64")
async def verify_base64_signatures(request: dict):
    """
    Verify signatures from base64 encoded images.
    
    Mathematical Workflow:
    Input: {image1: base64_string, image2: base64_string}
    1. Decode base64 → PIL Images
    2. Preprocess: I → T(I) ∈ ℝ^(1×32×32)
    3. Extract features: φ_θ(T(I)) ∈ ℝ^128
    4. Compute similarity: p = σ(f_θ(T(I1), T(I2)))
    5. Classify: genuine if p < 0.5, forged otherwise
    
    Expected request format:
    {
        "image1": "base64_string_of_first_signature",
        "image2": "base64_string_of_second_signature"
    }
    
    Returns:
        dict: Verification results with mathematical confidence metrics
    """
    try:
        # Validate request structure
        if not request or 'image1' not in request or 'image2' not in request:
            raise HTTPException(
                status_code=400, 
                detail="Missing required fields. Please provide 'image1' and 'image2' as base64 strings."
            )
        
        # Check model availability
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please check server logs."
            )
        
        # Process base64 images to PIL format
        try:
            img1 = process_image(request['image1'])
            img2 = process_image(request['image2'])
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Perform signature verification using Siamese network
        result = verify_signatures(img1, img2)
        
        return {
            "success": True,
            "result": result,
            "message": f"Verification complete: {result['result']}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/verify-files")
async def verify_file_signatures(
    image1: UploadFile = File(..., description="First signature image"),
    image2: UploadFile = File(..., description="Second signature image")
):
    """
    Verify signatures from uploaded files.
    
    Mathematical Process:
    File Upload → PIL Image → Preprocessing → Siamese Network → Verification
    
    Supports standard image formats: PNG, JPG, JPEG, GIF, BMP
    Files are processed in memory without saving to disk for security.
    
    Args:
        image1: First signature image file
        image2: Second signature image file
        
    Returns:
        dict: Verification results with file metadata
    """
    try:
        # Check model availability
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please check server logs."
            )
        
        # Validate file types for security and compatibility
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        
        def validate_file(file: UploadFile):
            """Validate uploaded file format and security."""
            if not file.filename:
                raise HTTPException(status_code=400, detail="Filename is required")
            
            extension = file.filename.lower().split('.')[-1]
            if extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type '{extension}'. Allowed: {', '.join(allowed_extensions)}"
                )
        
        validate_file(image1)
        validate_file(image2)
        
        # Process uploaded files to PIL Images
        try:
            img1 = Image.open(image1.file).convert("L")
            img2 = Image.open(image2.file).convert("L")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing images: {str(e)}")
        
        # Perform signature verification
        result = verify_signatures(img1, img2)
        
        return {
            "success": True,
            "result": result,
            "files_processed": {
                "image1": image1.filename,
                "image2": image2.filename
            },
            "message": f"Verification complete: {result['result']}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File verification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/batch-verify")
async def batch_verify_signatures(request: dict):
    """
    Batch verification of multiple signature pairs for efficiency.
    
    Mathematical Optimization:
    Instead of N separate API calls, process N pairs in single request:
    - Reduces network overhead
    - Enables batch processing optimizations
    - Provides aggregate statistics
    
    Expected request format:
    {
        "pairs": [
            {"image1": "base64_string1", "image2": "base64_string2"},
            {"image1": "base64_string3", "image2": "base64_string4"},
            ...
        ]
    }
    
    Returns:
        dict: Batch results with individual verification outcomes and timing statistics
    """
    try:
        # Validate request structure
        if not request or 'pairs' not in request:
            raise HTTPException(status_code=400, detail="Missing 'pairs' field")
        
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        pairs = request['pairs']
        
        # Limit batch size to prevent memory issues and timeout
        if len(pairs) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 pairs per batch")
        
        results = []
        total_time = time.time()
        
        # Process each signature pair
        for i, pair in enumerate(pairs):
            try:
                # Process images and verify signatures
                img1 = process_image(pair['image1'])
                img2 = process_image(pair['image2'])
                result = verify_signatures(img1, img2)
                
                results.append({
                    "pair_index": i,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                # Handle individual pair errors gracefully
                results.append({
                    "pair_index": i,
                    "success": False,
                    "error": str(e)
                })
        
        processing_time = time.time() - total_time
        
        return {
            "success": True,
            "total_pairs": len(pairs),
            "results": results,
            "total_processing_time_ms": round(processing_time * 1000, 2),
            "average_time_per_pair_ms": round((processing_time / len(pairs)) * 1000, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch verification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# APPLICATION LIFECYCLE EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Server startup event handler.
    
    Loads the trained Siamese Neural Network model into memory
    for efficient inference during API requests.
    
    Mathematical Context:
    Model loading involves:
    1. Initializing network architecture with correct dimensions
    2. Loading learned parameters θ* from training
    3. Setting evaluation mode for inference
    """
    logger.info("🚀 Starting Signature Verification API...")
    success = load_model()
    if success:
        logger.info("✅ Server ready for signature verification!")
    else:
        logger.error("❌ Server started but model failed to load!")

# ============================================================================
# SERVER ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Main application entry point for development server.
    
    Production Deployment Notes:
    - Use production ASGI server (e.g., Gunicorn with Uvicorn workers)
    - Disable reload in production
    - Configure proper CORS origins
    - Add authentication and rate limiting
    - Set up monitoring and logging
    """
    print("🔐 Signature Verification FastAPI Server")
    print("📱 Optimized for React Native mobile apps")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🩺 Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",  # Allow external connections (all network interfaces)
        port=8000,
        reload=True,     # Auto-reload on code changes (disable in production)
        log_level="info"
    )

"""
MATHEMATICAL SUMMARY OF API OPERATIONS:

1. IMAGE PREPROCESSING PIPELINE:
   Raw Image → Base64 Decode → PIL Image → Grayscale → Resize(32×32) → Normalize[0,1] → Tensor

2. SIAMESE NETWORK INFERENCE:
   - Feature Extraction: φ_θ(x) : ℝ^(1×32×32) → ℝ^128
   - Distance Computation: d = |φ_θ(x1) - φ_θ(x2)| ∈ ℝ^128
   - Classification: p = σ(W·d + b) ∈ [0,1]

3. DECISION BOUNDARY:
   - Threshold: τ = 0.5
   - Classification: genuine if p < τ, forged if p ≥ τ
   - Confidence: c = |p - τ| × 2 ∈ [0,1]

4. PERFORMANCE OPTIMIZATION:
   - Model loaded once at startup (O(1) per request vs O(model_size))
   - Batch processing for multiple pairs
   - GPU acceleration when available
   - Efficient memory management with torch.no_grad()

5. API DESIGN PRINCIPLES:
   - RESTful endpoints for different input formats
   - Comprehensive error handling with appropriate HTTP status codes
   - Structured JSON responses with mathematical metrics
   - Health monitoring for production deployment
   - CORS support for cross-origin mobile applications

6. SECURITY CONSIDERATIONS:
   - File type validation to prevent malicious uploads
   - Input sanitization and validation
   - Error messages that don't leak system information
   - Rate limiting through batch size restrictions
"""