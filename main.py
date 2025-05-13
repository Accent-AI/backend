from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import sys
import os
import traceback

# Ensure the correct path is added for importing model_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import classify_accent

app = FastAPI()

# Updated CORS configuration with explicit origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://accent-ai.github.io", 
        "https://accent-ai.github.io/frontend/", 
        "http://localhost:3000",  # For local development
        "http://localhost:5173"   # For Vite's default port
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    # Validate file type 
    print(f"Received file: {file.filename}, size: {file.size} bytes")
    
    # Validate file extension
    allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}")
    
    # Generate a unique temporary filename to avoid conflicts
    temp_path = os.path.join("temp", f"temp_{os.urandom(16).hex()}_{file.filename}")
    
    try:
        # Ensure temp directory exists
        os.makedirs("temp", exist_ok=True)
        
        # Save the uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate file was saved correctly
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        
        # Validate file size
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Classify the accent
        result = classify_accent(temp_path)
        
        return result
    
    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    
    except Exception as e:
        # Log the full error for server-side debugging
        print(f"Error processing file: {traceback.format_exc()}")
        
        # Raise an HTTP exception with a user-friendly message
        raise HTTPException(status_code=500, detail=f"Error processing the audio file: {str(e)}")
    
    finally:
        # Always attempt to remove the temporary file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as cleanup_error:
            print(f"Error cleaning up temporary file: {cleanup_error}")

# Add a simple endpoint for testing CORS
@app.get("/health")
async def health_check():
    return {"status": "ok", "cors": "enabled"}