from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import sys
import os
import traceback
import uuid
import glob
import tempfile
from pathlib import Path
import time
import wave
import struct
import ffmpeg

# Ensure correct path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the shared model and other modules
from shared_model import get_classifier, ACCENTS_EN
from model_utils import classify_accent
from accent_analyzer import analyze_accent_similarity, get_available_accents

# Import our new audio preprocessing module
from audio_preprocessing import preprocess_audio, cleanup_temp_files

app = FastAPI()
g = get_classifier()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=1200,
)

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    unique_id = None
    temp_dir = Path(tempfile.gettempdir()) / "accentify_temp"

    try:
        # Validate file type - accept more audio formats
        print(f"📁 Received file: {file.filename}, size: {file.size} bytes")
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}")

        # Create temp directory
        temp_dir.mkdir(exist_ok=True)
        
        # Generate unique filenames
        unique_id = uuid.uuid4().hex
        original_path = temp_dir / f"original_{unique_id}{Path(file.filename).suffix}"
        
        print(f"💾 Using original path: {original_path}")

        # Read and save file content
        content = await file.read()
        print(f"📖 Read {len(content)} bytes from uploaded file")
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Save original file
        with open(original_path, "wb") as buffer:
            buffer.write(content)
            buffer.flush()
            os.fsync(buffer.fileno())
        
        print(f"✅ Saved original file: {original_path}")
        
        # Verify file exists
        if not original_path.exists() or original_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Failed to save file properly")
        
        # Wait for Windows file system
        time.sleep(0.1)
        
        # NEW: Use shared preprocessing function
        print("🎵 Starting comprehensive audio preprocessing...")
        try:
            processed_path = preprocess_audio(str(original_path), temp_dir)
            print(f"✅ Audio preprocessing completed: {processed_path}")
        except Exception as preprocessing_error:
            print(f"❌ Audio preprocessing failed: {preprocessing_error}")
            raise HTTPException(status_code=500, detail=f"Audio preprocessing failed: {str(preprocessing_error)}")
        
        # Verify processed file
        final_path = Path(processed_path)
        if not final_path.exists() or final_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Processed audio file is invalid")
        
        print(f"🎯 Classifying with preprocessed audio: {processed_path}")
        
        # Classify accent using preprocessed audio
        result = classify_accent(processed_path)
        print("✅ Classification successful")
        return result

    except HTTPException:
        raise

    except Exception as e:
        print(f"❌ Error processing file:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing the audio file: {str(e)}")

    finally:
        # Clean up all temp files using the shared cleanup function
        if unique_id:
            cleanup_temp_files(unique_id, temp_dir)

@app.post("/classify_accent")
async def classify_accent_endpoint(
    file: UploadFile = File(...),
    target_accent: str = Form(...)
):
    """
    Analyze how much an audio file sounds like a specific accent
    
    Args:
        file: Audio file to analyze
        target_accent: The accent to compare against (e.g., "US", "England", "Australia")
    
    Returns:
        JSON with similarity analysis results
    """
    unique_id = None
    temp_dir = Path(tempfile.gettempdir()) / "accentify_temp"

    try:
        # Validate file type
        print(f"📁 Received file: {file.filename}, size: {file.size} bytes")
        print(f"🎯 Target accent: {target_accent}")
        
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}")

        # Validate target accent
        available_accents = get_available_accents()
        if target_accent not in available_accents:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid accent '{target_accent}'. Available accents: {available_accents}"
            )

        # Create temp directory
        temp_dir.mkdir(exist_ok=True)
        
        # Generate unique filenames
        unique_id = uuid.uuid4().hex
        original_path = temp_dir / f"original_{unique_id}{Path(file.filename).suffix}"
        
        print(f"💾 Using original path: {original_path}")

        # Read and save file content
        content = await file.read()
        print(f"📖 Read {len(content)} bytes from uploaded file")
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Save original file
        with open(original_path, "wb") as buffer:
            buffer.write(content)
            buffer.flush()
            os.fsync(buffer.fileno())
        
        print(f"✅ Saved original file: {original_path}")
        
        # Verify file exists
        if not original_path.exists() or original_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Failed to save file properly")
        
        # Wait for Windows file system
        time.sleep(0.1)
        
        # NEW: Use shared preprocessing function
        print("🎵 Starting comprehensive audio preprocessing...")
        try:
            processed_path = preprocess_audio(str(original_path), temp_dir)
            print(f"✅ Audio preprocessing completed: {processed_path}")
        except Exception as preprocessing_error:
            print(f"❌ Audio preprocessing failed: {preprocessing_error}")
            raise HTTPException(status_code=500, detail=f"Audio preprocessing failed: {str(preprocessing_error)}")
        
        # Verify processed file
        final_path = Path(processed_path)
        if not final_path.exists() or final_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Processed audio file is invalid")
        
        print(f"🎯 Analyzing accent similarity with preprocessed audio: {processed_path}")
        
        # Analyze accent similarity using preprocessed audio
        result = analyze_accent_similarity(processed_path, target_accent, use_randomizer=True)
        print("✅ Accent similarity analysis successful")
        return result

    except HTTPException:
        raise

    except Exception as e:
        print(f"❌ Error processing file:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing the audio file: {str(e)}")

    finally:
        # Clean up all temp files using the shared cleanup function
        if unique_id:
            cleanup_temp_files(unique_id, temp_dir)

@app.get("/available_accents")
async def get_accents():
    """Get list of available accents for analysis"""
    try:
        accents = get_available_accents()
        return {"accents": accents}
    except Exception as e:
        print(f"❌ Error getting available accents: {e}")
        # Fallback to hardcoded list if function fails
        return {"accents": ACCENTS_EN}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test if the shared classifier can be loaded
        classifier = get_classifier()
        return {
            "status": "ok", 
            "cors": "enabled",
            "model_loaded": True,
            "preprocessing": "enabled"
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e),
            "model_loaded": False,
            "preprocessing": "unknown"
        }