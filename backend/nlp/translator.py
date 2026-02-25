"""
Translation using the Sarvam AI REST API.

API endpoint: https://api.sarvam.ai/translate
Auth: Bearer token via SARVAM_API_KEY env var

Supported language codes:
    Hindi     → hi-IN
    Malayalam → ml-IN
    Tamil     → ta-IN
    Kannada   → kn-IN
    Marathi   → mr-IN
    English   → en-IN
"""
from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
REQUEST_TIMEOUT = 10  # seconds

# Languages that map to "en-IN" — no translation needed
ENGLISH_CODES = {"en-IN", "en", "en-US", "en-GB"}

SUPPORTED_LANG_CODES = {
    "hi-IN",
    "ml-IN",
    "ta-IN",
    "kn-IN",
    "mr-IN",
    "en-IN",
}


def _get_api_key() -> str | None:
    return os.getenv("SARVAM_API_KEY")


def _call_sarvam(text: str, source_lang: str, target_lang: str) -> str:
    """
    Make a single translation request to Sarvam AI.

    Returns the translated string or raises on HTTP / network error.
    """
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("SARVAM_API_KEY is not set in the environment.")

    payload = {
        "input": text,
        "source_language_code": source_lang,
        "target_language_code": target_lang,
        "speaker_gender": "Female",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": True,
    }

    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }

    response = requests.post(
        SARVAM_TRANSLATE_URL,
        json=payload,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    # Sarvam returns {"translated_text": "..."}
    translated = data.get("translated_text") or data.get("translation") or ""
    return translated.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate *text* from *source_lang* to English (en-IN).

    If *source_lang* is already English, or if the API call fails,
    returns the original *text* unchanged.

    Args:
        text: The text to translate.
        source_lang: ISO language code of the source text (e.g. "hi-IN").

    Returns:
        Translated English string, or original text on failure.
    """
    if not text or not text.strip():
        return text

    # Normalise: treat bare "en" variants as English
    if source_lang in ENGLISH_CODES:
        return text

    try:
        translated = _call_sarvam(text, source_lang=source_lang, target_lang="en-IN")
        if translated:
            logger.debug(
                "translate_to_english: %s -> en-IN | original=%r | translated=%r",
                source_lang,
                text[:60],
                translated[:60],
            )
            return translated
        # Empty response — return original
        return text

    except requests.exceptions.Timeout:
        logger.warning("Sarvam API timeout while translating to English (src=%s).", source_lang)
        return text
    except requests.exceptions.HTTPError as exc:
        logger.warning("Sarvam API HTTP error: %s", exc)
        return text
    except Exception as exc:
        logger.warning("Translation to English failed: %s", exc)
        return text


def translate_to_language(text: str, target_lang: str) -> str:
    """
    Translate *text* from English to *target_lang*.

    If *target_lang* is English, or if the API call fails,
    returns the original *text* unchanged.

    Args:
        text: English text to translate.
        target_lang: ISO language code of the desired output (e.g. "hi-IN").

    Returns:
        Translated string, or original text on failure.
    """
    if not text or not text.strip():
        return text

    if target_lang in ENGLISH_CODES:
        return text

    try:
        translated = _call_sarvam(text, source_lang="en-IN", target_lang=target_lang)
        if translated:
            logger.debug(
                "translate_to_language: en-IN -> %s | original=%r | translated=%r",
                target_lang,
                text[:60],
                translated[:60],
            )
            return translated
        return text

    except requests.exceptions.Timeout:
        logger.warning("Sarvam API timeout while translating to %s.", target_lang)
        return text
    except requests.exceptions.HTTPError as exc:
        logger.warning("Sarvam API HTTP error: %s", exc)
        return text
    except Exception as exc:
        logger.warning("Translation to %s failed: %s", target_lang, exc)
        return text
