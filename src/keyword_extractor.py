from __future__ import annotations

from typing import Dict, List

from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords_tfidf(text: str, top_n: int = 20) -> List[str]:
    # Extract top N unique terms using TF-IDF term weighting.
    if not text or not text.strip():
        return []

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    ranked = sorted(zip(feature_names, scores), key=lambda item: item[1], reverse=True)
    keywords = [term for term, score in ranked if score > 0]
    return keywords[:top_n]


def extract_keywords_rake(text: str, top_n: int = 20) -> List[str]:
    # Extract top N multi-word phrases using RAKE.
    if not text or not text.strip():
        return []

    rake = Rake()
    rake.extract_keywords_from_text(text)

    ranked_phrases = rake.get_ranked_phrases_with_scores()
    keywords = [phrase for _, phrase in ranked_phrases]
    return keywords[:top_n]


def extract_keywords(text: str, top_n: int = 20) -> List[str]:
    # Combine and deduplicate keyword extractions from TF-IDF and RAKE.
    tfidf_keywords = extract_keywords_tfidf(text, top_n=top_n)
    rake_keywords = extract_keywords_rake(text, top_n=top_n)

    # Preserve order while deduplicating between methods.
    seen = set()
    combined: List[str] = []
    for keyword in tfidf_keywords + rake_keywords:
        normalized = keyword.lower().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            combined.append(keyword)

    return combined[:top_n]


def find_missing_keywords(resume_text: str, jd_keywords: List[str]) -> List[str]:
    if not resume_text or not resume_text.strip():
        return jd_keywords

    resume_lower = resume_text.lower()
    missing: List[str] = []
    for keyword in jd_keywords:
        if keyword and keyword.lower() not in resume_lower:
            missing.append(keyword)

    return missing


def categorize_keywords(keywords: List[str]) -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {
        "Technical Skills": [],
        "Soft Skills": [],
        "Tools": [],
        "Qualifications": [],
    }

    technical_terms = {
        "python",
        "java",
        "javascript",
        "sql",
        "react",
        "node",
        "nlp",
        "machine learning",
        "deep learning",
        "data analysis",
        "api",
    }
    soft_terms = {
        "communication",
        "teamwork",
        "leadership",
        "problem solving",
        "adaptability",
        "time management",
        "critical thinking",
    }
    tool_terms = {
        "git",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "linux",
        "excel",
        "tableau",
        "power bi",
        "jira",
    }
    qualification_terms = {
        "bachelor",
        "master",
        "phd",
        "degree",
        "certification",
        "certified",
        "diploma",
    }

    for keyword in keywords:
        normalized = keyword.lower()
        if any(term in normalized for term in technical_terms):
            categories["Technical Skills"].append(keyword)
        elif any(term in normalized for term in soft_terms):
            categories["Soft Skills"].append(keyword)
        elif any(term in normalized for term in tool_terms):
            categories["Tools"].append(keyword)
        elif any(term in normalized for term in qualification_terms):
            categories["Qualifications"].append(keyword)
        else:
            categories["Technical Skills"].append(keyword)

    return categories
