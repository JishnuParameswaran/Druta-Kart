"""
Emotion analysis using HuggingFace transformers.

Model: j-hartmann/emotion-english-distilroberta-base
Outputs: anger, disgust, fear, joy, neutral, sadness, surprise

Handles Hinglish and Kanglish by running the model on the raw text —
the model is robust to code-mixed input as it uses subword tokenization.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Lazily loaded pipeline — avoids loading PyTorch at import time
_pipeline = None

EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

LABEL_MAP = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "neutral": "neutral",
    "sadness": "sadness",
    "surprise": "surprise",
}

# Emotions that warrant priority / escalation handling
HIGH_URGENCY_EMOTIONS = {"anger", "fear", "disgust"}


def _get_pipeline():
    """Load the emotion pipeline once and cache it for the process lifetime."""
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline  # noqa: PLC0415

            _pipeline = pipeline(
                "text-classification",
                model=EMOTION_MODEL,
                top_k=None,      # return all labels with scores
                truncation=True,
                max_length=512,
            )
            logger.info("Emotion analysis pipeline loaded: %s", EMOTION_MODEL)
        except Exception as exc:
            logger.error("Failed to load emotion pipeline: %s", exc)
            raise
    return _pipeline


def analyze_emotion(text: str) -> dict:
    """
    Analyse the dominant emotion in *text*.

    Works on plain English, Hinglish (Hindi+English), and Kanglish
    (Kannada+English) because the model uses subword BPE tokenisation and
    has seen code-mixed text during pre-training.

    Returns a dict with keys:
        label   — dominant emotion string (see LABEL_MAP)
        score   — confidence float [0, 1]
        all     — list of {label, score} for every emotion class
        urgent  — bool, True if the emotion warrants priority handling
    """
    if not text or not text.strip():
        return {
            "label": "neutral",
            "score": 1.0,
            "all": [{"label": "neutral", "score": 1.0}],
            "urgent": False,
        }

    try:
        pipe = _get_pipeline()
        results = pipe(text[:512])

        # pipeline with top_k=None returns [[{label, score}, ...]]
        all_scores = results[0] if isinstance(results[0], list) else results
        all_scores = sorted(all_scores, key=lambda x: x["score"], reverse=True)

        dominant = all_scores[0]
        label = LABEL_MAP.get(dominant["label"].lower(), dominant["label"].lower())

        return {
            "label": label,
            "score": round(dominant["score"], 4),
            "all": [
                {
                    "label": LABEL_MAP.get(r["label"].lower(), r["label"].lower()),
                    "score": round(r["score"], 4),
                }
                for r in all_scores
            ],
            "urgent": label in HIGH_URGENCY_EMOTIONS,
        }

    except Exception as exc:
        logger.warning("Emotion analysis failed for text=%r: %s", text[:80], exc)
        return {
            "label": "neutral",
            "score": 0.0,
            "all": [],
            "urgent": False,
            "error": str(exc),
        }


def is_urgent(text: str) -> bool:
    """Convenience wrapper — True if the text carries a high-urgency emotion."""
    return analyze_emotion(text).get("urgent", False)
