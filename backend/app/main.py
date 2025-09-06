"""
FastAPI Demo Wrapper for Signature Verification
Exposes endpoints for main.py functionality (quick test, auto-enroll, verification etc.)
"""

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ml.verification.signature_verifier import SignatureVerifier

# Initialize FastAPI app
app = FastAPI()

# Allow all origins for demo (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global verifier instance
verifier = SignatureVerifier()
verifier.load_or_create_model()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Signature Verification Demo API",
        "docs": "/docs",
        "available_endpoints": [
            "/quick-test",
            "/enrolled-users",
            "/enroll-user",
            "/verify-signature"
        ]
    }


@app.get("/quick-test")
async def quick_test():
    model_path = "ml/training/data/models/siamese_model.h5"
    users_dir = Path("ml/training/data/users")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="No trained model found. Run training first.")

    if not users_dir.exists() or not any(users_dir.iterdir()):
        raise HTTPException(status_code=400, detail="No user data found. Add signature images first.")

    enrolled_users = verifier.get_enrolled_users()
    return {"enrolled_users": enrolled_users}


@app.get("/enrolled-users")
async def get_enrolled_users():
    """List currently enrolled users"""
    return {"users": verifier.get_enrolled_users()}


@app.post("/enroll-user")
async def enroll_user(username: str = Form(...), files: List[UploadFile] = File(...)):
    """
    Enroll a user with multiple signatures
    - Requires minimum 3 images
    """
    if len(files) < 3:
        raise HTTPException(status_code=400, detail="At least 3 signatures required.")

    user_dir = Path(f"ml/training/data/users/{username}/train")
    user_dir.mkdir(parents=True, exist_ok=True)

    file_paths = []
    for i, file in enumerate(files):
        file_path = user_dir / f"sig_{i+1}{Path(file.filename).suffix}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        file_paths.append(str(file_path))

    try:
        verifier.enroll_user(username, file_paths)
        return {"message": f"User {username} enrolled successfully", "signatures": len(file_paths)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify-signature")
async def verify_signature(username: str = Form(...), file: UploadFile = File(...)):
    """
    Verify a single signature against enrolled user
    """
    test_dir = Path(f"ml/training/data/users/{username}/test")
    test_dir.mkdir(parents=True, exist_ok=True)

    file_path = test_dir / f"test_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        result = verifier.verify_signature(username, str(file_path))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
