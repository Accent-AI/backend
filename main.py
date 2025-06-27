from fastapi import FastAPI, UploadFile, File, HTTPException
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
        "http://localhost:5173",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

def validate_and_fix_wav(file_path):
    """Validate WAV file and attempt to fix common issues"""
    try:
        # Try to open with wave module first
        with wave.open(str(file_path), 'rb') as wf:
            frames = wf.getnframes()
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            
            print(f"📊 WAV file info:")
            print(f"   - Frames: {frames}")
            print(f"   - Sample rate: {sample_rate} Hz")
            print(f"   - Channels: {channels}")
            print(f"   - Sample width: {sample_width} bytes")
            
            if frames == 0:
                raise ValueError("WAV file has no audio data")
            
            # Check if it's a valid format for speech processing
            if sample_rate < 8000:
                print("⚠️ Warning: Sample rate is very low, this might affect classification")
            
            return True
            
    except wave.Error as e:
        print(f"❌ WAV validation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error validating WAV: {e}")
        return False

def convert_to_wav_with_ffmpeg(input_path, output_path):
    """Convert audio file to WAV format using ffmpeg"""
    try:
        print(f"🔄 Converting to WAV using ffmpeg: {input_path} -> {output_path}")
        
        # Use ffmpeg to convert to standard WAV format
        (
            ffmpeg
            .input(str(input_path))
            .output(
                str(output_path),
                acodec='pcm_s16le',  # 16-bit PCM
                ar=16000,            # 16kHz sample rate
                ac=1,                # mono
                f='wav'              # WAV format
            )
            .overwrite_output()
            .run(quiet=True, capture_stdout=True)
        )
        
        print(f"✅ Successfully converted to WAV: {output_path}")
        return True
        
    except ffmpeg.Error as e:
        print(f"❌ FFmpeg conversion failed: {e}")
        print(f"   stdout: {e.stdout.decode() if e.stdout else 'None'}")
        print(f"   stderr: {e.stderr.decode() if e.stderr else 'None'}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during conversion: {e}")
        return False

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    unique_id = None

    try:
        # Validate file type - now accept more audio formats
        print(f"📁 Received file: {file.filename}, size: {file.size} bytes")
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}")

        # Use system temp directory
        temp_dir = Path(tempfile.gettempdir()) / "accentify_temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Generate unique filenames
        unique_id = uuid.uuid4().hex
        original_path = temp_dir / f"original_{unique_id}{Path(file.filename).suffix}"
        converted_path = temp_dir / f"converted_{unique_id}.wav"
        
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
        
        # Determine which file to use for classification
        classification_path = None
        
        # If it's already a WAV file, validate it
        if file.filename.lower().endswith('.wav'):
            if validate_and_fix_wav(original_path):
                print("✅ Original WAV file is valid")
                classification_path = str(original_path)
            else:
                print("⚠️ Original WAV file is invalid, converting with ffmpeg...")
                if convert_to_wav_with_ffmpeg(original_path, converted_path):
                    classification_path = str(converted_path)
                else:
                    raise HTTPException(status_code=400, detail="Unable to process WAV file")
        else:
            # For non-WAV files, always convert
            print(f"🔄 Converting {Path(file.filename).suffix} file to WAV...")
            if convert_to_wav_with_ffmpeg(original_path, converted_path):
                classification_path = str(converted_path)
            else:
                raise HTTPException(status_code=400, detail=f"Unable to convert {Path(file.filename).suffix} file to WAV")
        
        if not classification_path:
            raise HTTPException(status_code=500, detail="No valid audio file available for classification")
        
        # Final validation of the file to be used
        final_path = Path(classification_path)
        if not final_path.exists() or final_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Processed audio file is invalid")
        
        # Additional wait to ensure file is ready
        time.sleep(0.1)
        
        print(f"🎯 Classifying with path: {classification_path}")
        
        # Classify accent
        result = classify_accent(classification_path)
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
        # Clean up all temp files
        temp_dir = Path(tempfile.gettempdir()) / "accentify_temp"
        if temp_dir.exists():
            for temp_file in temp_dir.glob(f"*{unique_id}*"):
                try:
                    temp_file.unlink()
                    print(f"🧹 Deleted temporary file: {temp_file}")
                except Exception as cleanup_error:
                    print(f"❌ Error deleting temp file: {cleanup_error}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "cors": "enabled"}