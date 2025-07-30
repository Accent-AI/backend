# from speechbrain.pretrained.interfaces import foreign_class
from speechbrain.pretrained import EncoderClassifier
import torch
import os

# Global classifier instance to be shared across modules
_classifier = None

def get_classifier():
    """Get or create the shared classifier instance"""
    global _classifier
    if _classifier is None:
        print("🔄 Loading SpeechBrain classifier...")
        try:
            # _classifier = foreign_class(
            #     source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
            #     pymodule_file="custom_interface.py",
            #     classname="CustomEncoderWav2vec2Classifier"
            # )
            _classifier = EncoderClassifier.from_hparams(
                source="Jzuluaga/accent-id-commonaccent_ecapa",
                savedir="pretrained_models/accent-id-commonaccent_ecapa"
            )
            print("✅ SpeechBrain classifier loaded successfully")
        except Exception as e:
            print(f"❌ Error loading SpeechBrain classifier: {e}")
            raise
    return _classifier

def clear_classifier():
    """Clear the classifier from memory (for testing/debugging)"""
    global _classifier
    if _classifier is not None:
        del _classifier
        _classifier = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("🧹 Classifier cleared from memory")

# Available accents list
# ACCENTS_EN = [
#     'US', 'England', 'Australia', 'Indian', 'Canada', 'Bermuda', 'Scotland',
#     'African', 'Ireland', 'New Zealand', 'Wales', 'Malaysia', 'Philippines',
#     'Singapore', 'Hong Kong', 'Southatlandtic'
# ] 

ACCENTS_EN = ['England', 'US', 'Canada', 'Australia', 'Indian', 'Scotland', 'Ireland',
              'African', 'Malaysia', 'New Zealand', 'Southatlandtic', 'Bermuda',
              'Philippines', 'Hong Kong', 'Wales', 'Singapore']