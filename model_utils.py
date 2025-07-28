import torchaudio
import torch
import numpy as np
import os
import requests
import json
from typing import Dict, Any
from shared_model import get_classifier, ACCENTS_EN
from ai_prompts import get_accent_message_prompt
from ai_api import call_groq_api

def generate_accent_message(accent_result: Dict[str, Any]) -> str:
    """Generate a descriptive message about the accent classification using Groq API"""
    try:
        # Get the top 3 accents by probability
        accent_names = accent_result["accents"]
        probabilities = accent_result["probabilities"]
        
        # Create accent-probability pairs and sort by probability
        accent_prob_pairs = list(zip(accent_names, probabilities))
        top_accents = sorted(accent_prob_pairs, key=lambda x: x[1], reverse=True)[:3]
        
        # Format the prompt for the AI
        detected_accent = accent_result["accent"]
        confidence = accent_result["score"]
        
        prompt = get_accent_message_prompt(detected_accent, confidence, top_accents)

        # Try Groq API first (free and fast)
        groq_response = call_groq_api(prompt)
        if groq_response:
            return groq_response
        
        # Fallback to Hugging Face API
        hf_response = call_huggingface_api(prompt)
        if hf_response:
            return hf_response
        
        # Final fallback - generate a simple message
        return generate_fallback_message(detected_accent, confidence, top_accents)
        
    except Exception as e:
        print(f"❌ Failed to generate accent message: {e}")
        # Return fallback message
        return generate_fallback_message(
            accent_result.get("accent", "Unknown"), 
            accent_result.get("score", 0),
            []
        )

def call_huggingface_api(prompt: str) -> str:
    """Call Hugging Face Inference API as fallback"""
    try:
        # Using a free model from Hugging Face
        # No API key required for public models, but rate limited
        url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_length": 150,
                "temperature": 0.7,
                "do_sample": True
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                message = result[0].get("generated_text", "").strip()
                # Clean up the response (remove the original prompt)
                if prompt in message:
                    message = message.replace(prompt, "").strip()
                
                if message:
                    print("✅ Generated message using Hugging Face API")
                    return message
        else:
            print(f"❌ Hugging Face API error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Hugging Face API call failed: {e}")
        return None

def generate_fallback_message(accent: str, confidence: float, top_accents: list) -> str:
    """Generate a simple fallback message when APIs are unavailable"""
    confidence_desc = "high" if confidence > 0.7 else "moderate" if confidence > 0.5 else "low"
    
    accent_display = accent.title() if accent else "Unknown"
    
    messages = [
        f"I detected a {accent_display} accent with {confidence_desc} confidence ({confidence:.1%}). This suggests you likely have linguistic influences from {accent_display} English patterns.",
        f"Your speech patterns most closely match a {accent_display} accent (confidence: {confidence:.1%}). The analysis shows {confidence_desc} certainty in this classification.",
        f"Based on the audio analysis, I identified a {accent_display} accent with {confidence:.1%} confidence. This indicates {confidence_desc} similarity to {accent_display} speech patterns."
    ]
    
    import random
    return random.choice(messages)

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
        
        # Get the shared classifier
        classifier = get_classifier()
        
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

        # Create the base result
        result = {
            "accent": text_lab[0],  # text_lab is now a list
            "probabilities": formatted_probs,
            "score": formatted_score,
            "accents": ACCENTS_EN
        }
        
        # Generate AI message
        print("🤖 Generating AI message...")
        ai_message = generate_accent_message(result)
        result["message"] = ai_message
        
        return result
        
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        print(f"   File path: {file_path}")
        print(f"   File exists: {os.path.exists(file_path)}")
        print(f"   File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'}")
        
        # Fallback: try the original method one more time
        try:
            print("🔄 Attempting fallback with original classify_file method...")
            classifier = get_classifier()
            out_prob, score, index, text_lab = classifier.classify_file(file_path)
            
            formatted_probs = [round(float(p), 4) for sublist in out_prob.tolist() for p in sublist]
            formatted_score = round(float(score), 4)
            
            result = {
                "accent": text_lab,
                "probabilities": formatted_probs,
                "score": formatted_score,
                "accents": ACCENTS_EN
            }
            
            # Generate AI message for fallback too
            ai_message = generate_accent_message(result)
            result["message"] = ai_message
            
            return result
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            raise e  # Raise the original error