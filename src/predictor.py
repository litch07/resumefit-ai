import re
from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import hstack

def clean_resume_text(text: str) -> str:
    """Clean resume text identically to the training pipeline."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_classifier() -> tuple:
    """
    Load the trained LinearSVC model, the vectorizers dictionary, 
    and the label encoder from the models/ directory.
    Returns (model, vectorizers_dict, label_encoder) or (None, None, None) on failure.
    """
    models_dir = Path("models")
    
    model_path = models_dir / "job_classifier.pkl"
    vec_path = models_dir / "tfidf_vectorizer.pkl"
    le_path = models_dir / "label_encoder.pkl"
    
    if not (model_path.exists() and vec_path.exists() and le_path.exists()):
        print("Warning: One or more classification model artifacts are missing in the 'models/' directory.")
        return None, None, None
        
    try:
        model = joblib.load(model_path)
        vectorizers_dict = joblib.load(vec_path)
        label_encoder = joblib.load(le_path)
        return model, vectorizers_dict, label_encoder
    except Exception as e:
        print(f"Error loading classifier artifacts: {e}")
        return None, None, None

def predict_job_roles(resume_text: str, model, vectorizers: dict, label_encoder, top_n: int = 3) -> list:
    """
    Predict the top_n most likely job roles for a given resume text.
    Returns a list of dictionaries with the predicted 'role' and 'confidence' (0-100).
    """
    if not resume_text or not str(resume_text).strip():
        return []
        
    if model is None or vectorizers is None or label_encoder is None:
        return []
        
    try:
        cleaned_text = clean_resume_text(resume_text)
        
        word_features = vectorizers["word"].transform([cleaned_text])
        char_features = vectorizers["char"].transform([cleaned_text])
        
        # Combine word and character features into a unified sparse matrix.
        features = hstack([word_features, char_features])
        
        # Get raw decision scores from LinearSVC.
        decision_scores = model.decision_function(features)[0]
        
        # Convert raw scores to 0-100% confidence using numerically stable Softmax.
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probabilities = exp_scores / exp_scores.sum()
        
        top_indices = np.argsort(probabilities)[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            role = label_encoder.inverse_transform([idx])[0]
            confidence = round(float(probabilities[idx]) * 100, 1)
            
            results.append({
                "role": role,
                "confidence": confidence
            })
            
        return results
    except Exception as e:
        print(f"Error during job role prediction: {e}")
        return []
