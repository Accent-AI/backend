import torchaudio
import torch
import numpy as np
import os
import requests
import json
import random
from typing import Dict, Any, Tuple
from shared_model import get_classifier, ACCENTS_EN
from ai_prompts import get_similarity_message_prompt
from ai_api import call_groq_api

def randomize_out_prob(out_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """
    Apply small randomization to probability outputs and recalculate results.
    
    Args:
        out_prob: Original probability tensor from classifier
        
    Returns:
        Tuple of (modified_out_prob, score, index, text_lab)
    """
    # Clone the tensor to avoid modifying the original
    modified_prob = out_prob.clone()
    
    # Apply randomization (± up to 0.05) to each probability value
    for i in range(modified_prob.shape[1]):  # Iterate through probability values
        # Generate random value between -0.05 and +0.05
        random_offset = (random.random() - 0.5) * 0.1  # Range: -0.05 to +0.05
        
        # Apply offset and ensure value stays in [0, 1] range
        new_value = modified_prob[0, i].item() + random_offset
        new_value = max(0.0, min(1.0, new_value))  # Clamp to [0, 1]
        
        # Round to 4 decimal places
        modified_prob[0, i] = round(new_value, 4)
    
    # Find the new maximum value and its index
    max_value, max_index = torch.max(modified_prob, dim=1)
    
    # Get the corresponding accent label
    accent_index = max_index.item()
    accent_label = ACCENTS_EN[accent_index]
    
    return modified_prob, max_value, max_index, [accent_label]

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

def analyze_accent_with_randomizer(file_path: str) -> Dict[str, Any]:
    """
    Analyze accent from audio file with randomized probability output.
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        Dictionary containing analysis results with randomized probabilities
    """
    try:
        print(f"🎯 Starting accent analysis with randomizer for: {file_path}")
        
        # Get the shared classifier
        classifier = get_classifier()
        
        # Preprocess the audio file
        waveform, sample_rate = preprocess_audio(file_path)
        
        # Add batch dimension if needed
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add batch dimension: [1, time]
        
        # Use the classifier's classify_batch method with the preprocessed tensor
        with torch.no_grad():
            # Get the length tensor (required by SpeechBrain)
            wav_lens = torch.tensor([1.0])  # Full length
            
            # Perform classification
            original_out_prob, original_score, original_index, original_text_lab = classifier.classify_batch(waveform, wav_lens)
            print(f"Original classification results:\n"
                  f"  out_prob: {original_out_prob}\n"
                  f"  score: {original_score}\n"
                  f"  index: {original_index}\n"
                  f"  text_lab: {original_text_lab}")
            
            # Apply randomization
            out_prob, score, index, text_lab = randomize_out_prob(original_out_prob)
            print(f"Randomized classification results:\n"
                  f"  out_prob: {out_prob}\n"
                  f"  score: {score}\n"
                  f"  index: {index}\n"
                  f"  text_lab: {text_lab}")
        
        # Format results
        formatted_probs = [round(float(p), 4) for p in out_prob[0].tolist()]
        formatted_score = round(float(score[0]), 4)
        
        # Get the top 3 accents for comparison
        accent_prob_pairs = list(zip(ACCENTS_EN, formatted_probs))
        top_accents = sorted(accent_prob_pairs, key=lambda x: x[1], reverse=True)[:3]
        
        print(f"✅ Analysis completed:")
        print(f"   - Top detected accent: {text_lab[0]} ({formatted_score:.2%})")
        print(f"   - Top 3 accents: {[(acc, f'{prob:.2%}') for acc, prob in top_accents[:3]]}")

        # Create the result
        result = {
            "detected_accent": text_lab[0],
            "detected_confidence": formatted_score,
            "all_probabilities": dict(zip(ACCENTS_EN, formatted_probs)),
            "top_3_accents": [{"accent": acc, "probability": prob} for acc, prob in top_accents],
            "original_results": {
                "detected_accent": original_text_lab[0],
                "detected_confidence": round(float(original_score[0]), 4),
                "all_probabilities": dict(zip(ACCENTS_EN, [round(float(p), 4) for p in original_out_prob[0].tolist()]))
            }
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Accent analysis with randomizer failed: {e}")
        print(f"   File path: {file_path}")
        print(f"   File exists: {os.path.exists(file_path)}")
        print(f"   File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'}")
        raise

def analyze_accent_similarity(file_path: str, target_accent: str, use_randomizer: bool = False) -> Dict[str, Any]:
    """
    Analyze how much an audio file sounds like a specific accent
    
    Args:
        file_path: Path to the audio file
        target_accent: The accent to compare against (must be in ACCENTS_EN)
        use_randomizer: Whether to apply randomization to probabilities
    
    Returns:
        Dictionary containing analysis results
    """
    try:
        print(f"🎯 Starting accent similarity analysis for: {file_path}")
        print(f"🎯 Target accent: {target_accent}")
        print(f"🎯 Using randomizer: {use_randomizer}")
        
        # Validate target accent
        if target_accent not in ACCENTS_EN:
            raise ValueError(f"Invalid accent '{target_accent}'. Available accents: {ACCENTS_EN}")
        
        # Get the shared classifier
        classifier = get_classifier()
        
        # Preprocess the audio file
        waveform, sample_rate = preprocess_audio(file_path)
        
        # Add batch dimension if needed
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add batch dimension: [1, time]
        
        # Use the classifier's classify_batch method with the preprocessed tensor
        with torch.no_grad():
            # Get the length tensor (required by SpeechBrain)
            wav_lens = torch.tensor([1.0])  # Full length
            
            # Perform classification
            original_out_prob, original_score, original_index, original_text_lab = classifier.classify_batch(waveform, wav_lens)
            print(f"Classification results:\n"
                  f"  original_out_prob: {original_out_prob}\n"
                  f"  original_score: {original_score}\n"
                  f"  original_index: {original_index}\n"
                  f"  original_text_lab: {original_text_lab}")
            # Apply randomization if requested
            if use_randomizer:
                out_prob, score, index, text_lab = randomize_out_prob(original_out_prob)
                print(f"Using randomized classification results")
            else:
                out_prob, score, index, text_lab = original_out_prob, original_score, original_index, original_text_lab
                print(f"Using original classification results")
            
            print(f"Classification results:\n"
                  f"  out_prob: {out_prob}\n"
                  f"  score: {score}\n"
                  f"  index: {index}\n"
                  f"  text_lab: {text_lab}")
        
        # Format results
        formatted_probs = [round(float(p), 4) for p in out_prob[0].tolist()]
        formatted_score = round(float(score[0]), 4)
        
        # Find the probability for the target accent
        target_accent_index = ACCENTS_EN.index(target_accent)
        target_accent_probability = formatted_probs[target_accent_index]
        
        # Get the top 3 accents for comparison
        accent_prob_pairs = list(zip(ACCENTS_EN, formatted_probs))
        top_accents = sorted(accent_prob_pairs, key=lambda x: x[1], reverse=True)[:3]
        
        # Determine similarity level
        similarity_level = get_similarity_level(target_accent_probability)
        
        print(f"✅ Analysis completed:")
        print(f"   - Target accent ({target_accent}): {target_accent_probability:.2%}")
        print(f"   - Similarity level: {similarity_level}")
        print(f"   - Top detected accent: {text_lab[0]} ({formatted_score:.2%})")

        # Create the result
        result = {
            "target_accent": target_accent,
            "target_accent_probability": target_accent_probability,
            "similarity_level": similarity_level,
            "similarity_percentage": round(target_accent_probability * 100, 2),
            "detected_accent": text_lab[0],
            "detected_confidence": formatted_score,
            "all_probabilities": dict(zip(ACCENTS_EN, formatted_probs)),
            "top_3_accents": [{"accent": acc, "probability": prob} for acc, prob in top_accents],
            "message": generate_similarity_message(target_accent, target_accent_probability, similarity_level, text_lab[0], formatted_score),
            "randomized": use_randomizer
        }
        
        # Include original results if randomizer was used
        if use_randomizer:
            original_formatted_probs = [round(float(p), 4) for p in original_out_prob[0].tolist()]
            original_target_prob = original_formatted_probs[target_accent_index]
            result["original_results"] = {
                "target_accent_probability": original_target_prob,
                "detected_accent": original_text_lab[0],
                "detected_confidence": round(float(original_score[0]), 4),
                "all_probabilities": dict(zip(ACCENTS_EN, original_formatted_probs))
            }
        
        return result
        
    except Exception as e:
        print(f"❌ Accent similarity analysis failed: {e}")
        print(f"   File path: {file_path}")
        print(f"   Target accent: {target_accent}")
        print(f"   File exists: {os.path.exists(file_path)}")
        print(f"   File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'}")
        raise

def get_similarity_level(probability: float) -> str:
    """Determine similarity level based on probability"""
    if probability >= 0.8:
        return "Very High"
    elif probability >= 0.6:
        return "High"
    elif probability >= 0.4:
        return "Moderate"
    elif probability >= 0.2:
        return "Low"
    else:
        return "Very Low"

def generate_similarity_message(target_accent: str, target_prob: float, similarity_level: str, detected_accent: str, detected_confidence: float) -> str:
    """Generate a descriptive message about accent similarity, using Groq API if available"""
    target_accent_display = target_accent.title()
    detected_accent_display = detected_accent.title()
    prompt = get_similarity_message_prompt(target_accent_display, similarity_level, target_prob, detected_accent_display, detected_confidence)
    groq_response = call_groq_api(prompt)
    if groq_response:
        return groq_response
    # Fallback to the original random message logic
    if similarity_level in ["Very High", "High"]:
        messages = [
            f"Your speech shows {similarity_level.lower()} similarity to {target_accent_display} accent patterns ({target_prob:.1%}). This suggests strong linguistic influences from {target_accent_display} English.",
            f"The analysis indicates {similarity_level.lower()} resemblance to {target_accent_display} accent ({target_prob:.1%}). Your speech patterns closely match {target_accent_display} pronunciation characteristics.",
            f"Your audio demonstrates {similarity_level.lower()} similarity to {target_accent_display} accent ({target_prob:.1%}). This reflects significant {target_accent_display} English speech patterns."
        ]
    elif similarity_level == "Moderate":
        messages = [
            f"Your speech shows moderate similarity to {target_accent_display} accent ({target_prob:.1%}). There are some {target_accent_display} English influences present.",
            f"The analysis reveals moderate resemblance to {target_accent_display} accent patterns ({target_prob:.1%}). Some {target_accent_display} pronunciation characteristics are detectable.",
            f"Your audio demonstrates moderate similarity to {target_accent_display} accent ({target_prob:.1%}). This indicates some {target_accent_display} English speech influences."
        ]
    else:  # Low or Very Low
        messages = [
            f"Your speech shows {similarity_level.lower()} similarity to {target_accent_display} accent ({target_prob:.1%}). Limited {target_accent_display} English influences are detected.",
            f"The analysis indicates {similarity_level.lower()} resemblance to {target_accent_display} accent patterns ({target_prob:.1%}). Few {target_accent_display} pronunciation characteristics are present.",
            f"Your audio demonstrates {similarity_level.lower()} similarity to {target_accent_display} accent ({target_prob:.1%}). This suggests minimal {target_accent_display} English speech patterns."
        ]
    if detected_accent != target_accent:
        comparison = f" Note: The system detected {detected_accent_display} accent as the primary match ({detected_confidence:.1%} confidence)."
    else:
        comparison = f" The system also identified {detected_accent_display} as the primary accent match ({detected_confidence:.1%} confidence)."
    base_message = random.choice(messages)
    return base_message + comparison

def get_available_accents() -> list:
    """Get list of available accents for analysis"""
    return ACCENTS_EN.copy()