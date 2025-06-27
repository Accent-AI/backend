from speechbrain.pretrained import EncoderClassifier
import torchaudio
import torch
import numpy as np
import os

ACCENTS_EN = ['England', 'US', 'Canada', 'Australia', 'Indian', 'Scotland', 'Ireland',
              'African', 'Malaysia', 'New Zealand', 'Southatlandtic', 'Bermuda',
              'Philippines', 'Hong Kong', 'Wales', 'Singapore']

classifier = EncoderClassifier.from_hparams(
    source="Jzuluaga/accent-id-commonaccent_ecapa",
    savedir="pretrained_models/accent-id-commonaccent_ecapa"
)

def preprocess_audio(file_path):
    """Preprocess audio file to ensure compatibility"""
    try:
        print(f"🔧 Preprocessing audio file: {file_path}")
        
        # Check if file exists and is readable
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Cannot read audio file: {file_path}")
        
        # Try to load with torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
        print(f"✅ Loaded audio: shape={waveform.shape}, sr={sample_rate}")
        
        # Basic validation
        if waveform.size(0) == 0 or waveform.size(1) == 0:
            raise ValueError("Audio file appears to be empty")
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print("🔧 Converted stereo to mono")
        
        # Resample if necessary (SpeechBrain models typically expect 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
            print(f"🔧 Resampled to 16000Hz")
        
        # Normalize audio
        if torch.max(torch.abs(waveform)) > 0:
            waveform = waveform / torch.max(torch.abs(waveform))
        
        # Ensure the tensor is in the right format for SpeechBrain
        # SpeechBrain expects [batch_size, time] format
        if waveform.dim() == 2 and waveform.size(0) == 1:
            waveform = waveform.squeeze(0)  # Remove channel dimension for mono
        
        print(f"✅ Preprocessed audio shape: {waveform.shape}")
        return waveform, sample_rate
        
    except Exception as e:
        print(f"❌ Audio preprocessing failed: {e}")
        raise

def classify_accent(file_path):
    try:
        print(f"🎯 Starting accent classification for: {file_path}")
        
        # Preprocess the audio file
        waveform, sample_rate = preprocess_audio(file_path)
        
        # Instead of using classify_file, use classify_batch with the preprocessed audio
        # This bypasses the file loading issue
        print("🔧 Using direct tensor classification to bypass file loading...")
        
        # Add batch dimension if needed
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add batch dimension: [1, time]
        
        # Use the classifier's classify_batch method with the preprocessed tensor
        with torch.no_grad():
            # Get the length tensor (required by SpeechBrain)
            wav_lens = torch.tensor([1.0])  # Full length
            
            # Perform classification
            out_prob, score, index, text_lab = classifier.classify_batch(waveform, wav_lens)
        
        # Format results
        formatted_probs = [round(float(p), 4) for sublist in out_prob.tolist() for p in sublist]
        formatted_score = round(float(score[0]), 4)  # score is now a tensor
        
        print(f"✅ Classification completed: {text_lab[0]} (confidence: {formatted_score})")

        return {
            "accent": text_lab[0],  # text_lab is now a list
            "probabilities": formatted_probs,
            "score": formatted_score,
            "accents": ACCENTS_EN
        }
        
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        print(f"   File path: {file_path}")
        print(f"   File exists: {os.path.exists(file_path)}")
        print(f"   File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'}")
        
        # Fallback: try the original method one more time
        try:
            print("🔄 Attempting fallback with original classify_file method...")
            out_prob, score, index, text_lab = classifier.classify_file(file_path)
            
            formatted_probs = [round(float(p), 4) for sublist in out_prob.tolist() for p in sublist]
            formatted_score = round(float(score), 4)
            
            return {
                "accent": text_lab,
                "probabilities": formatted_probs,
                "score": formatted_score,
                "accents": ACCENTS_EN
            }
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            raise e  # Raise the original error