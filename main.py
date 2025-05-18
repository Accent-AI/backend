from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import sys
import os
import traceback
import uuid
import glob
import ffmpeg

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
    # Absolute path for temp directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEMP_DIR = os.path.join(BASE_DIR, "temp")
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Track all temporary files for cleanup
    temp_files = []

    # Validate file type
    print(f"Received file: {file.filename}, size: {file.size} bytes")
    allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}")

    # Generate unique filenames for temporary files
    unique_id = uuid.uuid4().hex
    original_temp_path = os.path.join(TEMP_DIR, f"temp_{unique_id}_original_{file.filename}")
    wav_temp_path = os.path.join(TEMP_DIR, f"temp_{unique_id}_converted.wav")
    
    temp_files = []  # Track all temp files for cleanup

    try:
        # Save uploaded file
        with open(original_temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        temp_files.append(original_temp_path)
        print(f"✅ Saved original temp file to: {original_temp_path}")

        # Check file exists and isn't empty
        if not os.path.exists(original_temp_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        file_size = os.path.getsize(original_temp_path)
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
            
        # Convert audio to WAV format using ffmpeg
        try:
            print(f"🔄 Converting {original_temp_path} to WAV format...")
            (
                ffmpeg
                .input(original_temp_path)
                .output(wav_temp_path, acodec='pcm_s16le', ac=1, ar='16k')
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            temp_files.append(wav_temp_path)
            print(f"✅ Converted to WAV: {wav_temp_path}")
            
            # Verify the converted file exists and isn't empty
            if not os.path.exists(wav_temp_path):
                raise HTTPException(status_code=500, detail="Failed to convert audio to WAV format")
            wav_file_size = os.path.getsize(wav_temp_path)
            if wav_file_size == 0:
                raise HTTPException(status_code=500, detail="Converted WAV file is empty")
                
        except ffmpeg.Error as e:
            print(f"❌ FFmpeg error: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            raise HTTPException(status_code=500, detail=f"Error converting audio: {str(e)}")

        # Classify accent using the WAV file
        result = classify_accent(wav_temp_path)
        return result

    except HTTPException:
        raise

    except Exception as e:
        print(f"❌ Error processing file: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing the audio file: {str(e)}")

    finally:
        try:
            file.file.close()
            
            # Clean up all temporary files created during this request
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"🧹 Deleted temporary file: {temp_file}")
                    
            # Clean up any lingering temp files in the temp directory
            for temp_file in glob.glob(os.path.join(TEMP_DIR, "temp_*")):
                try:
                    os.remove(temp_file)
                    print(f"🧹 Deleted lingering temporary file in temp dir: {temp_file}")
                except Exception as temp_del_error:
                    print(f"❌ Failed to delete lingering temp file {temp_file}: {temp_del_error}")
                
            # Additionally clean up any lingering temp files in the same directory as main.py
            main_dir = os.path.dirname(os.path.abspath(__file__))
            temp_pattern = os.path.join(main_dir, "temp_*")
            for temp_file in glob.glob(temp_pattern):
                try:
                    os.remove(temp_file)
                    print(f"🧹 Deleted lingering temporary file in main dir: {temp_file}")
                except Exception as temp_del_error:
                    print(f"❌ Failed to delete lingering temporary file {temp_file}: {temp_del_error}")
        except Exception as cleanup_error:
            print(f"❌ Error cleaning up temporary files: {cleanup_error}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "cors": "enabled"}