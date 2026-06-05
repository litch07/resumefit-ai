import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_MODEL_CACHE = None

def load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        print("Loading SBERT model...")
        _MODEL_CACHE = SentenceTransformer("all-MiniLM-L6-v2")
        print("SBERT model ready.")
    return _MODEL_CACHE

def get_embedding(text, model):
    if not text or not str(text).strip():
        # Return a zero vector for empty inputs
        return np.zeros(384)
    return model.encode(text)

def get_similarity_score(text1, text2, model):
    if not text1 or not str(text1).strip() or not text2 or not str(text2).strip():
        return 0.0
        
    emb1 = get_embedding(text1, model)
    emb2 = get_embedding(text2, model)
    
    # Wrap embeddings in 2D arrays for sklearn compatibility.
    sim = cosine_similarity([emb1], [emb2])[0][0]
    
    # Scale cosine similarity [-1, 1] to a [0, 100] percentage.
    score = max(0.0, float(sim) * 100)
    return round(score, 1)
