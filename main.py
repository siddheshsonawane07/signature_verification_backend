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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Signature Verification API",
    description="AI-powered signature verification using Siamese Neural Networks",
    version="1.0.0"
)

# Enable CORS for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your mobile app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Global model variable
model = None

def load_model():
    """Load the trained model"""
    global model
    try:
        model = snn().to(device)
        if os.path.exists('best_model.pth'):
            model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
            logger.info("✅ Loaded best_model.pth")
        elif os.path.exists('model_last.pth'):
            model.load_state_dict(torch.load('model_last.pth', map_location=device, weights_only=True))
            logger.info("✅ Loaded model_last.pth")
        else:
            raise FileNotFoundError("❌ No trained model found. Please train the model first.")
        model.eval()
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False

def process_image(image_data):
    """Process image from various input formats"""
    try:
        if isinstance(image_data, str):
            # Base64 string
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("L")
        elif hasattr(image_data, 'read'):
            # File upload
            image = Image.open(image_data).convert("L")
        else:
            raise ValueError("Unsupported image format")
        
        return image
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def verify_signatures(img1, img2):
    """Compare two signature images using the trained model"""
    try:
        start_time = time.time()
        
        # Apply transformations
        img1_tensor = transform(img1).unsqueeze(0).to(device)
        img2_tensor = transform(img2).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            logit = model(img1_tensor, img2_tensor)
            similarity_score = torch.sigmoid(logit).item()
        
        processing_time = time.time() - start_time
        
        # Interpret result (lower score = more similar = genuine)
        is_genuine = similarity_score < 0.5
        confidence = abs(similarity_score - 0.5) * 2  # Convert to 0-1 confidence scale
        
        return {
            'is_genuine': is_genuine,
            'similarity_score': float(similarity_score),
            'confidence': float(confidence),
            'result': 'Genuine' if is_genuine else 'Forged',
            'processing_time_ms': round(processing_time * 1000, 2),
            'threshold': 0.5
        }
    except Exception as e:
        raise RuntimeError(f"Error during verification: {str(e)}")

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
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
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device),
        "timestamp": time.time()
    }

@app.post("/verify-base64")
async def verify_base64_signatures(request: dict):
    """
    Verify signatures from base64 encoded images
    
    Expected request format:
    {
        "image1": "base64_string_of_first_signature",
        "image2": "base64_string_of_second_signature"
    }
    """
    try:
        # Validate request
        if not request or 'image1' not in request or 'image2' not in request:
            raise HTTPException(
                status_code=400, 
                detail="Missing required fields. Please provide 'image1' and 'image2' as base64 strings."
            )
        
        # Check if model is loaded
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please check server logs."
            )
        
        # Process images
        try:
            img1 = process_image(request['image1'])
            img2 = process_image(request['image2'])
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Verify signatures
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
    Verify signatures from uploaded files
    
    Accepts: PNG, JPG, JPEG, GIF, BMP files
    """
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please check server logs."
            )
        
        # Validate file types
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        
        def validate_file(file: UploadFile):
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
        
        # Process images
        try:
            img1 = Image.open(image1.file).convert("L")
            img2 = Image.open(image2.file).convert("L")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing images: {str(e)}")
        
        # Verify signatures
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
    Batch verification of multiple signature pairs
    
    Expected request format:
    {
        "pairs": [
            {"image1": "base64_string1", "image2": "base64_string2"},
            {"image1": "base64_string3", "image2": "base64_string4"},
            ...
        ]
    }
    """
    try:
        if not request or 'pairs' not in request:
            raise HTTPException(status_code=400, detail="Missing 'pairs' field")
        
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        pairs = request['pairs']
        if len(pairs) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 10 pairs per batch")
        
        results = []
        total_time = time.time()
        
        for i, pair in enumerate(pairs):
            try:
                img1 = process_image(pair['image1'])
                img2 = process_image(pair['image2'])
                result = verify_signatures(img1, img2)
                
                results.append({
                    "pair_index": i,
                    "success": True,
                    "result": result
                })
            except Exception as e:
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

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model when server starts"""
    logger.info("🚀 Starting Signature Verification API...")
    success = load_model()
    if success:
        logger.info("✅ Server ready for signature verification!")
    else:
        logger.error("❌ Server started but model failed to load!")

# Main entry point
if __name__ == "__main__":
    print("🔐 Signature Verification FastAPI Server")
    print("📱 Optimized for React Native mobile apps")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🩺 Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",  # Allow external connections
        port=8000,
        reload=True,     # Auto-reload on code changes (disable in production)
        log_level="info"
    )