"""
Language detection using Groq LLM (llama-3.3-70b-versatile).

Replaces lingua-language-detector. Handles all Indian languages and
code-mixed variants (Hinglish, Kanglish, Manglish, etc.) perfectly.

Returns: en-IN, hi-IN, ta-IN, kn-IN, ml-IN, mr-IN, te-IN,
         hinglish, kanglish, manglish
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_VALID_CODES = {
    "en-IN", "hi-IN", "ta-IN", "kn-IN",
    "ml-IN", "mr-IN", "te-IN",
    "hinglish", "kanglish", "manglish",
}

# Code-mixed variants the LLM understands natively — no translation needed
_ENGLISH_LIKE = {"en-IN", "hinglish", "kanglish", "manglish"}


def detect_language(text: str) -> str:
    """Detect language of text using Groq LLM. Returns a language code."""
    if not text or not text.strip():
        return "en-IN"
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
        prompt = (
            "Detect the language of this message.\n"
            "Reply with ONLY the language code, nothing else.\n"
            "Options: en-IN, hi-IN, ta-IN, kn-IN, "
            "ml-IN, mr-IN, hinglish, kanglish, "
            "manglish, te-IN\n"
            f'Message: "{text}"\n'
            "Language code:"
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        code = resp.choices[0].message.content.strip().lower()
        if code in _VALID_CODES:
            return code
        for valid in _VALID_CODES:
            if valid in code:
                return valid
        logger.warning("LLM returned unknown language code '%s', defaulting to en-IN", code)
        return "en-IN"
    except Exception as exc:
        logger.warning("LLM language detection failed: %s", exc)
        return "en-IN"


def is_english(text: str) -> bool:
    """Return True if text is English or a code-mixed variant (Hinglish/Kanglish/Manglish)
    that the LLM understands natively without translation."""
    return detect_language(text) in _ENGLISH_LIKE
