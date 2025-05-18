from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import sys
import os
import traceback

# Ensure correct path for importing model_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_utils import classify_accent

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://accent-ai.github.io",
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    temp_path = None

    # Absolute path for temp directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEMP_DIR = os.path.join(BASE_DIR, "temp")
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Validate file type
    print(f"Received file: {file.filename}, size: {file.size} bytes")
    allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}")

    temp_path = os.path.join(TEMP_DIR, f"temp_{os.urandom(16).hex()}_{file.filename}")

    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"✅ Saved temp file to: {temp_path}")

        # Check file exists and isn't empty
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Classify accent
        result = classify_accent(temp_path)
        return result

    except HTTPException:
        raise

    except Exception as e:
        print(f"❌ Error processing file: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing the audio file: {str(e)}")

    finally:
        try:
            file.file.close()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"🧹 Deleted temporary file: {temp_path}")
            else:
                print(f"⚠️ Temp file not deleted or didn't exist: {temp_path}")
        except Exception as cleanup_error:
            print(f"❌ Error cleaning up temporary file: {cleanup_error}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "cors": "enabled"}
