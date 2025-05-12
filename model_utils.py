
from speechbrain.pretrained import EncoderClassifier

ACCENTS_EN = ['England', 'US', 'Canada', 'Australia', 'Indian', 'Scotland', 'Ireland',
              'African', 'Malaysia', 'New Zealand', 'Southatlandtic', 'Bermuda',
              'Philippines', 'Hong Kong', 'Wales', 'Singapore']

classifier = EncoderClassifier.from_hparams(
    source="Jzuluaga/accent-id-commonaccent_ecapa",
    savedir="pretrained_models/accent-id-commonaccent_ecapa"
)

def classify_accent(file_path):
    out_prob, score, index, text_lab = classifier.classify_file(file_path)
    formatted_probs = [round(float(p), 4) for sublist in out_prob.tolist() for p in sublist]
    formatted_score = round(float(score), 4)

    return {
        "accent": text_lab,
        "probabilities": formatted_probs,
        "score": formatted_score,
        "accents": ACCENTS_EN
    }
