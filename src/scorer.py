from __future__ import annotations

from typing import List, Dict

from .embedder import get_similarity_score
from .keyword_extractor import extract_keywords, find_missing_keywords
from .preprocessor import preprocess, clean_text


def _count_words(text: str) -> int:
    if not text or not text.strip():
        return 0

    cleaned = clean_text(text)
    if not cleaned:
        return 0

    return len(cleaned.split())


def _score_label(score: float) -> str:
    if score > 80:
        return "Excellent"
    if score > 60:
        return "Good"
    if score > 40:
        return "Fair"
    return "Needs Work"


def _score_color(score: float) -> str:
    if score > 80:
        return "#059669"  # green
    if score > 60:
        return "#0EA5E9"  # teal
    if score > 40:
        return "#F59E0B"  # amber
    return "#EF4444"  # red


def generate_suggestions(missing_keywords: List[str]) -> List[str]:
    if not missing_keywords:
        return ["Great job! Your resume already covers the key job requirements."]

    suggestions: List[str] = []
    for keyword in missing_keywords[:5]:
        keyword_clean = keyword.strip()
        if not keyword_clean:
            continue

        suggestions.append(
            f"Consider adding {keyword_clean} to your skills or experience sections"
        )

    if not suggestions:
        return ["Consider expanding your resume with relevant skills from the job post."]

    return suggestions


def analyze(resume_text: str, job_description_text: str, model) -> Dict[str, object]:
    """
    Analyze resume vs. job description and return score, keywords, and suggestions.
    """
    # Orchestrate analysis: semantic scoring, keyword extraction, and gap identification.
    resume_clean = preprocess(resume_text)
    jd_clean = preprocess(job_description_text)

    # Compute SBERT semantic similarity score.
    score = get_similarity_score(resume_clean, jd_clean, model)

    jd_keywords = extract_keywords(job_description_text, top_n=20)
    missing_keywords = find_missing_keywords(resume_text or "", jd_keywords)
    suggestions = generate_suggestions(missing_keywords)

    return {
        "score": score,
        "score_label": _score_label(score),
        "score_color": _score_color(score),
        "missing_keywords": missing_keywords,
        "jd_keywords": jd_keywords,
        "suggestions": suggestions[:5],
        "resume_word_count": _count_words(resume_text or ""),
        "jd_word_count": _count_words(job_description_text or ""),
    }
