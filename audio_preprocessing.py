"""
Ultra-Fast Audio Preprocessing for Accent Classification
- Optimized noise gating with vectorized operations
- Reduced memory allocations and faster processing
- Minimal NumPy operations for maximum speed
"""

import io
import uuid
import tempfile
import warnings
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

warnings.filterwarnings("ignore", category=UserWarning)


def preprocess_audio(input_path: str, output_dir: Path = None) -> str:
    """
    Ultra-fast audio preprocessing pipeline:
    1. In-memory conversion to WAV (mono, 22050Hz)
    2. Ultra-fast vectorized noise gating
    3. Fast VAD-based speaker isolation
    4. Normalization and post-processing
    5. Save to file

    Args:
        input_path (str): Path to input audio file
        output_dir (Path, optional): Directory to save processed file

    Returns:
        str: Path to processed audio file
    """
    try:
        print(f"🎧 Starting preprocessing: {input_path}")

        # Setup temp dir
        if output_dir is None:
            output_dir = Path(tempfile.gettempdir()) / "accentify_temp"
        output_dir.mkdir(parents=True, exist_ok=True)

        unique_id = uuid.uuid4().hex
        processed_path = output_dir / f"processed_{unique_id}.wav"

        # Step 1: Load and convert audio in memory
        print("⚡ Loading audio (ffmpeg in-memory)...")
        audio_data, sr = _load_audio_fast(input_path)

        # Step 2: Ultra-fast vectorized noise gating
        print("🔇 Applying ultra-fast noise gate...")
        audio_data = _reduce_noise_ultra_fast(audio_data)

        # Step 3: VAD-based speaker isolation (optimized)
        print("🎙️ Isolating speaker with fast VAD...")
        audio_data = _isolate_primary_speaker_fast(audio_data, sr)

        # Step 4: Normalize and fade
        print("🧼 Normalizing and final processing...")
        audio_data = _post_process_audio(audio_data, sr)

        # Step 5: Save output
        sf.write(str(processed_path), audio_data, sr)
        print(f"✅ Saved to: {processed_path}")

        return str(processed_path)

    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        raise


def _load_audio_fast(input_path: str, sr: int = 22050):
    """Load and convert audio in memory using ffmpeg"""
    try:
        command = [
            "ffmpeg",
            "-i", input_path,
            "-f", "wav",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", str(sr),
            "-hide_banner",
            "-loglevel", "error",
            "-"
        ]
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {err.decode()}")

        audio_io = io.BytesIO(out)
        audio_data, sample_rate = sf.read(audio_io)

        return audio_data, sample_rate

    except Exception as e:
        print(f"❌ Failed to load audio: {e}")
        raise


def _reduce_noise_ultra_fast(audio: np.ndarray) -> np.ndarray:
    """
    Ultra-fast noise gating using vectorized operations
    - Pre-compute absolute values once
    - Use fast percentile approximation
    - Single vectorized masking operation
    """
    try:
        if len(audio) == 0:
            return audio
            
        # Single computation of absolute values
        abs_audio = np.abs(audio)
        
        # Fast threshold estimation using numpy's optimized percentile
        # Python 3.8 compatible version
        threshold = np.percentile(abs_audio, 25)
        
        # Ultra-fast vectorized gating - single operation
        # Create boolean mask and apply in one step
        mask = abs_audio >= threshold
        return audio * mask
        
    except Exception as e:
        print(f"⚠️ Noise gate failed: {e}")
        return audio


def _reduce_noise_alternative_ultra_fast(audio: np.ndarray) -> np.ndarray:
    """
    Alternative ultra-fast approach using sorted sampling for even more speed
    """
    try:
        if len(audio) == 0:
            return audio
            
        # Sample-based threshold estimation for very long audio files
        if len(audio) > 100000:  # For files longer than ~4.5 seconds at 22050Hz
            # Sample every nth element for threshold calculation
            sample_step = len(audio) // 10000
            sample_indices = slice(0, None, sample_step)
            threshold = np.percentile(np.abs(audio[sample_indices]), 25)
        else:
            # Full calculation for shorter files
            threshold = np.percentile(np.abs(audio), 25)
        
        # Single vectorized operation
        return np.where(np.abs(audio) >= threshold, audio, 0)
        
    except Exception as e:
        print(f"⚠️ Alternative noise gate failed: {e}")
        return audio


def _isolate_primary_speaker_fast(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Optimized VAD-based speaker isolation
    - Reduced frame calculations
    - More efficient concatenation
    """
    try:
        if len(audio) == 0:
            return audio
            
        # Use larger frames for faster processing
        frame_len = int(0.05 * sr)  # Increased from 0.03 to 0.05
        hop_len = frame_len // 4    # Larger hops for speed
        
        # More efficient RMS calculation
        rms = librosa.feature.rms(
            y=audio, 
            frame_length=frame_len, 
            hop_length=hop_len,
            center=False  # Disable centering for speed
        )[0]
        
        # Fast threshold using median instead of percentile for speed
        threshold = np.median(rms) * 0.5  # Equivalent to roughly 30th percentile
        
        # Pre-allocate list with estimated size
        segments = []
        segments_reserve = max(1, len(rms) // 4)  # Reserve space
        
        # Vectorized segment detection
        active_frames = rms > threshold
        
        # Find contiguous segments more efficiently
        if np.any(active_frames):
            # Use numpy to find transitions
            transitions = np.diff(np.concatenate(([False], active_frames, [False])).astype(int))
            starts = np.where(transitions == 1)[0]
            ends = np.where(transitions == -1)[0]
            
            for start_frame, end_frame in zip(starts, ends):
                start_sample = start_frame * hop_len
                end_sample = min((end_frame + 1) * hop_len, len(audio))
                if end_sample > start_sample:
                    segments.append(audio[start_sample:end_sample])
        
        if segments:
            return np.concatenate(segments)
        return audio

    except Exception as e:
        print(f"⚠️ Speaker isolation failed: {e}")
        return audio


def _post_process_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Optimized final normalization, fade-in/out, and cleanup
    """
    try:
        if len(audio) == 0:
            return audio
            
        # Fast DC removal
        audio = audio - np.mean(audio)
        
        # Fast normalization
        max_amp = np.max(np.abs(audio))
        if max_amp > 1e-8:  # Avoid division by very small numbers
            audio = 0.85 * audio / max_amp

        # Optimized fade calculation
        fade_len = min(int(0.01 * sr), len(audio) // 10)
        if fade_len > 1:
            # Pre-compute fade curves
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            
            # Apply fades
            audio[:fade_len] *= fade_in
            audio[-fade_len:] *= fade_out

        # Fast cleanup with explicit bounds
        return np.clip(audio, -0.85, 0.85)

    except Exception as e:
        print(f"⚠️ Post-processing failed: {e}")
        return audio


def cleanup_temp_files(unique_id: str, temp_dir: Path):
    """Clean up temporary files"""
    try:
        import glob
        pattern = str(temp_dir / f"*{unique_id}*")
        for file_path in glob.glob(pattern):
            try:
                Path(file_path).unlink()
                print(f"🗑️ Cleaned up: {file_path}")
            except Exception as e:
                print(f"⚠️ Could not delete {file_path}: {e}")
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")


# Additional ultra-fast alternative for Step 2 if needed
def _reduce_noise_minimal(audio: np.ndarray) -> np.ndarray:
    """
    Minimal noise reduction - fastest possible approach
    Uses simple energy-based detection with pre-computed statistics
    """
    try:
        if len(audio) == 0:
            return audio
            
        # Single pass statistics
        abs_audio = np.abs(audio)
        mean_energy = np.mean(abs_audio)
        
        # Simple threshold - much faster than percentile
        threshold = mean_energy * 0.3
        
        # Single boolean operation
        return audio * (abs_audio >= threshold)
        
    except Exception as e:
        print(f"⚠️ Minimal noise gate failed: {e}")
        return audio