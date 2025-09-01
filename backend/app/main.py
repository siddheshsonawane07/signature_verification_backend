"""
FastAPI Backend for Signature Verification System
Handles API endpoints, authentication, and ML service integration
"""

import os
import sys
import shutil
from datetime import datetime, timedelta
from typing import List, Optional
import asyncio
import json

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func

# Authentication imports
from passlib.context import CryptContext
from jose import JWTError, jwt

# Pydantic models
from pydantic import BaseModel, validator
import uuid

# Add ML module to path
sys.path.append('../ml')
from ml.verification.signature_verifier import SignatureVerificationEngine

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Signature Verification API",
    description="Person-specific signature verification using Siamese neural networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security setup
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Database setup
DATABASE_URL = "sqlite:///./data/signatures.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class UserDB(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    signature_count = Column(Integer, default=0)

class VerificationLogDB(Base):
    __tablename__ = "verification_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), nullable=False)
    verification_result = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    similarity_score = Column(Float, nullable=False)
    threshold_used = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    signature_path = Column(String(255))

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models
class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3 or len(v) > 50:
            raise ValueError('Username must be between 3 and 50 characters')
        if not v.isalnum():
            raise ValueError('Username must contain only alphanumeric characters')
        return v.lower()

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class VerificationResult(BaseModel):
    verified: bool
    confidence: float
    max_similarity: float
    average_similarity: float
    threshold: float
    cnn_similarity: float
    cnn_verified: bool
    methods_agree: bool
    timestamp: str
    signature_count_used: int

class EnrollmentResult(BaseModel):
    message: str
    profile_created: bool
    signature_count: int
    username: str
    enrollment_date: str

# Dependency functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Initialize ML service
ml_engine = SignatureVerificationEngine()

# Utility functions
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# API Endpoints
@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Advanced Signature Verification API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/health"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "signature-verification-api",
        "timestamp": datetime.utcnow().isoformat(),
        "ml_engine_status": "loaded" if ml_engine.feature_extractor is not None else "not_loaded"
    }

# Authentication endpoints
@app.post("/api/auth/register", response_model=dict)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register new user"""
    
    # Check if user already exists
    existing_user = db.query(UserDB).filter(UserDB.username == user_data.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    try:
        # Create user directory structure
        user_dir = f"../data/users/{user_data.username}"
        os.makedirs(f"{user_dir}/train", exist_ok=True)
        os.makedirs(f"{user_dir}/test", exist_ok=True)
        
        # Hash password and create user
        hashed_password = hash_password(user_data.password)
        db_user = UserDB(
            username=user_data.username,
            email=user_data.email,
            password_hash=hashed_password
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        return {
            "message": "User registered successfully",
            "username": user_data.username,
            "user_id": db_user.id
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/api/auth/login", response_model=Token)
async def login_user(user_data: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and return JWT token"""
    
    # Find user
    user = db.query(UserDB).filter(UserDB.username == user_data.username).first()
    if not user or not verify_password(user_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user account")
    
    # Create access token
    access_token = create_access_token(data={"sub": user.username})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_HOURS * 3600
    }

# User management endpoints
@app.get("/api/users/me")
async def get_current_user(username: str = Depends(verify_token), db: Session = Depends(get_db)):
    """Get current user information"""
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get user statistics
    stats = ml_engine.get_user_statistics(username)
    
    return {
        "username": user.username,
        "email": user.email,
        "created_at": user.created_at,
        "signature_count": user.signature_count,
        "is_enrolled": stats.get("signature_count", 0) > 0,
        "last_verification": stats.get("last_verification"),
        "verification_count": stats.get("verification_count", 0)
    }

@app.post("/api/users/{username}/enroll", response_model=EnrollmentResult)
async def enroll_user_signatures(
    username: str,
    signatures: List[UploadFile] = File(...),
    current_user: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Enroll user signatures for verification"""
    
    # Check authorization (users can only enroll themselves)
    if current_user != username:
        raise HTTPException(status_code=403, detail="Can only enroll your own signatures")
    
    # Validate signature count
    if len(signatures) < 5:
        raise HTTPException(status_code=400, detail="Minimum 5 signatures required for enrollment")
    
    if len(signatures) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 signatures allowed for enrollment")
    
    try:
        # Save signatures to train directory
        train_dir = f"../data/users/{username}/train"
        signature_paths = []
        
        # Clear existing training signatures
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(train_dir, exist_ok=True)
        
        # Save new signatures
        for i, signature in enumerate(signatures):
            # Validate file type
            if not signature.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {signature.filename} is not an image")
            
            # Generate unique filename
            file_extension = os.path.splitext(signature.filename)[1] or '.png'
            file_path = f"{train_dir}/signature_{i+1}_{uuid.uuid4().hex[:8]}{file_extension}"
            
            # Save file
            with open(file_path, "wb") as buffer:
                content = await signature.read()
                buffer.write(content)
            
            signature_paths.append(file_path)
        
        # Create signature profile using ML engine
        profile = ml_engine.create_user_profile(username, signature_paths)
        
        # Update user signature count in database
        user = db.query(UserDB).filter(UserDB.username == username).first()
        if user:
            user.signature_count = profile['signature_count']
            db.commit()
        
        return EnrollmentResult(
            message="Enrollment completed successfully",
            profile_created=True,
            signature_count=profile['signature_count'],
            username=username,
            enrollment_date=profile['enrollment_date']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")

@app.post("/api/users/{username}/verify", response_model=VerificationResult)
async def verify_user_signature(
    username: str,
    signature: UploadFile = File(...),
    current_user: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Verify signature against user's enrolled profile"""
    
    # Check authorization
    if current_user != username:
        raise HTTPException(status_code=403, detail="Can only verify your own signatures")
    
    # Validate file
    if not signature.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save test signature
        test_dir = f"../data/users/{username}/test"
        os.makedirs(test_dir, exist_ok=True)
        
        # Generate unique filename
        file_extension = os.path.splitext(signature.filename)[1] or '.png'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"{test_dir}/test_{timestamp}_{uuid.uuid4().hex[:8]}{file_extension}"
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await signature.read()
            buffer.write(content)
        
        # Perform verification
        result = ml_engine.verify_signature(username, file_path)
        
        # Log verification attempt
        log_entry = VerificationLogDB(
            username=username,
            verification_result=result['verified'],
            confidence=result['confidence'],
            similarity_score=result['max_similarity'],
            threshold_used=result['threshold'],
            signature_path=file_path
        )
        
        db.add(log_entry)
        db.commit()
        
        # Clean up test file after verification (optional)
        # os.remove(file_path)
        
        return VerificationResult(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

# Analytics endpoints
@app.get("/api/users/{username}/statistics")
async def get_user_statistics(
    username: str,
    current_user: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get user verification statistics"""
    
    if current_user != username:
        raise HTTPException(status_code=403, detail="Can only view your own statistics")
    
    try:
        # Get ML engine statistics
        ml_stats = ml_engine.get_user_statistics(username)
        
        # Get database statistics
        verification_logs = db.query(VerificationLogDB).filter(
            VerificationLogDB.username == username
        ).all()
        
        if verification_logs:
            recent_logs = [log for log in verification_logs 
                          if log.timestamp > datetime.utcnow() - timedelta(days=30)]
            
            db_stats = {
                'total_verifications': len(verification_logs),
                'recent_verifications': len(recent_logs),
                'success_rate': sum(1 for log in verification_logs if log.verification_result) / len(verification_logs),
                'average_confidence': sum(log.confidence for log in verification_logs) / len(verification_logs),
                'last_verification': max(log.timestamp for log in verification_logs).isoformat()
            }
        else:
            db_stats = {
                'total_verifications': 0,
                'recent_verifications': 0,
                'success_rate': 0,
                'average_confidence': 0,
                'last_verification': None
            }
        
        # Combine statistics
        combined_stats = {**ml_stats, **db_stats}
        return combined_stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.get("/api/admin/system-stats")
async def get_system_statistics(current_user: str = Depends(verify_token), db: Session = Depends(get_db)):
    """Get system-wide statistics (admin only)"""
    
    # In a real implementation, you'd check admin privileges here
    
    try:
        # Get database statistics
        total_users = db.query(UserDB).count()
        active_users = db.query(UserDB).filter(UserDB.is_active == True).count()
        total_verifications = db.query(VerificationLogDB).count()
        
        # Recent activity (last 7 days)
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        recent_verifications = db.query(VerificationLogDB).filter(
            VerificationLogDB.timestamp > recent_cutoff
        ).count()
        
        # Success rate
        successful_verifications = db.query(VerificationLogDB).filter(
            VerificationLogDB.verification_result == True
        ).count()
        
        success_rate = successful_verifications / total_verifications if total_verifications > 0 else 0
        
        # Get enrolled users
        enrolled_users = len(ml_engine.list_enrolled_users())
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'enrolled_users': enrolled_users,
            'total_verifications': total_verifications,
            'recent_verifications': recent_verifications,
            'overall_success_rate': success_rate,
            'system_uptime': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system statistics: {str(e)}")

# File management endpoints
@app.get("/api/users/{username}/signatures")
async def list_user_signatures(
    username: str,
    current_user: str = Depends(verify_token)
):
    """List user's enrolled signatures"""
    
    if current_user != username:
        raise HTTPException(status_code=403, detail="Can only view your own signatures")
    
    train_dir = f"../data/users/{username}/train"
    test_dir = f"../data/users/{username}/test"
    
    signatures = {
        'training_signatures': [],
        'test_signatures': []
    }
    
    # List training signatures
    if os.path.exists(train_dir):
        for filename in os.listdir(train_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = f"{train_dir}/{filename}"
                file_stats = os.stat(file_path)
                signatures['training_signatures'].append({
                    'filename': filename,
                    'size_bytes': file_stats.st_size,
                    'created_at': datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                })
    
    # List test signatures
    if os.path.exists(test_dir):
        for filename in os.listdir(test_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = f"{test_dir}/{filename}"
                file_stats = os.stat(file_path)
                signatures['test_signatures'].append({
                    'filename': filename,
                    'size_bytes': file_stats.st_size,
                    'created_at': datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                })
    
    return signatures

@app.delete("/api/users/{username}/profile")
async def delete_user_profile(
    username: str,
    current_user: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Delete user profile and all associated data"""
    
    if current_user != username:
        raise HTTPException(status_code=403, detail="Can only delete your own profile")
    
    try:
        # Delete ML profile
        profile_deleted = ml_engine.delete_user_profile(username)
        
        # Delete user directory
        user_dir = f"../data/users/{username}"
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
        
        # Delete verification logs
        db.query(VerificationLogDB).filter(VerificationLogDB.username == username).delete()
        
        # Delete user from database
        user = db.query(UserDB).filter(UserDB.username == username).first()
        if user:
            db.delete(user)
            db.commit()
        
        return {
            "message": "Profile deleted successfully",
            "username": username,
            "ml_profile_deleted": profile_deleted
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete profile: {str(e)}")

# Model management endpoints
@app.get("/api/model/info")
async def get_model_info(current_user: str = Depends(verify_token)):
    """Get information about the current ML model"""
    
    try:
        model_info = {
            'model_loaded': ml_engine.feature_extractor is not None,
            'model_path': ml_engine.model_path,
            'input_shape': ml_engine.siamese_network.input_shape,
            'feature_size': 256,
            'backbone': 'EfficientNetB3'
        }
        
        # Check if model files exist
        model_files = {
            'siamese_model': os.path.exists(f"{ml_engine.model_path}/siamese_model.h5"),
            'feature_extractor': os.path.exists(f"{ml_engine.model_path}/feature_extractor.h5"),
            'metadata': os.path.exists(f"{ml_engine.model_path}/model_metadata.json")
        }
        
        model_info['files_present'] = model_files
        
        return model_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

# Utility endpoints
@app.post("/api/utils/validate-image")
async def validate_image_quality(
    image: UploadFile = File(...),
    current_user: str = Depends(verify_token)
):
    """Validate image quality for signature verification"""
    
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save temporary file
        temp_path = f"../data/temp_{uuid.uuid4().hex}.png"
        with open(temp_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # Validate image quality
        is_valid, message = ml_engine.image_processor.validate_image_quality(temp_path)
        
        # Get detailed quality analysis
        from ml.verification.signature_verifier import VerificationUtils
        quality_analysis = VerificationUtils.analyze_signature_quality(temp_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return {
            'is_valid': is_valid,
            'message': message,
            'quality_analysis': quality_analysis
        }
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        
        raise HTTPException(status_code=500, detail=f"Image validation failed: {str(e)}")

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "File not found"}
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Signature Verification API...")
    
    # Ensure data directories exist
    os.makedirs("../data/users", exist_ok=True)
    os.makedirs("../data/models", exist_ok=True)
    os.makedirs("../data/profiles", exist_ok=True)
    
    # Initialize ML engine
    try:
        # This will load existing models or create new ones
        ml_engine._initialize_model()
        logger.info("ML engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ML engine: {e}")
    
    logger.info("API startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Signature Verification API...")

# Development and testing endpoints
if os.getenv("ENVIRONMENT") == "development":
    
    @app.post("/api/dev/create-test-data")
    async def create_test_data():
        """Create test data for development (development only)"""
        
        test_users = ["dev_user_1", "dev_user_2", "test_user"]
        
        for username in test_users:
            user_dir = f"../data/users/{username}"
            os.makedirs(f"{user_dir}/train", exist_ok=True)
            os.makedirs(f"{user_dir}/test", exist_ok=True)
            
            # Create placeholder files
            for i in range(5):
                placeholder_path = f"{user_dir}/train/placeholder_{i+1}.txt"
                with open(placeholder_path, 'w') as f:
                    f.write(f"Placeholder for signature {i+1} - replace with actual signature image")
        
        return {"message": f"Test data structure created for {len(test_users)} users"}
    
    @app.get("/api/dev/system-info")
    async def get_system_info():
        """Get system information (development only)"""
        
        import psutil
        import platform
        
        return {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'disk_usage': {
                'total_gb': round(psutil.disk_usage('.').total / (1024**3), 2),
                'free_gb': round(psutil.disk_usage('.').free / (1024**3), 2)
            },
            'tensorflow_version': tf.__version__ if 'tf' in globals() else 'Not loaded'
        }

# Main application runner
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Signature Verification API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--env', choices=['development', 'production'], default='development')
    
    args = parser.parse_args()
    
    # Set environment
    os.environ["ENVIRONMENT"] = args.env
    
    # Configure logging based on environment
    if args.env == "production":
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
    
    print(f"Starting Signature Verification API...")
    print(f"Environment: {args.env}")
    print(f"Host: {args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info" if args.env == "development" else "warning"
    )