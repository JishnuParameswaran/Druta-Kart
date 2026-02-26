"""
Druta Kart - Text-to-Speech Processor.

Uses Sarvam API (bulbul:v1) for Indian-language TTS.
Returns raw WAV bytes which the caller can stream directly to the client.
"""
from __future__ import annotations

import base64
import logging

from config import settings

logger = logging.getLogger(__name__)

# Sarvam TTS endpoint
_SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"

# Default voice; Sarvam supports "meera" (F), "pavithra" (F), "maitreyi" (F),
# "arvind" (M), "amol" (M) — all available for most Indian language codes.
_DEFAULT_SPEAKER = "meera"
_DEFAULT_MODEL = "bulbul:v1"

# Sarvam TTS has a ~500-char input limit per request
_MAX_CHARS = 500


def synthesize_speech(text: str, language: str = "en-IN") -> bytes:
    """Convert *text* to speech audio using Sarvam API.

    Args:
        text:     Text to synthesise (truncated to 500 chars if longer).
        language: Target language code (e.g. "hi-IN", "ta-IN", "en-IN").

    Returns:
        Raw WAV audio bytes, or empty bytes on failure.
    """
    if not text or not text.strip():
        return b""

    if not settings.sarvam_api_key:
        logger.warning("TTS: SARVAM_API_KEY not configured — skipping synthesis.")
        return b""

    text = text.strip()[:_MAX_CHARS]

    try:
        import requests  # lazy import

        response = requests.post(
            _SARVAM_TTS_URL,
            headers={"api-subscription-key": settings.sarvam_api_key},
            json={
                "inputs": [text],
                "target_language_code": language,
                "speaker": _DEFAULT_SPEAKER,
                "model": _DEFAULT_MODEL,
                "enable_preprocessing": True,
            },
            timeout=15,
        )
        response.raise_for_status()

        data = response.json()
        audio_b64: str = (data.get("audios") or [""])[0]
        if not audio_b64:
            logger.warning("TTS: empty audio in Sarvam response for lang=%s", language)
            return b""

        audio_bytes = base64.b64decode(audio_b64)
        logger.info(
            "TTS: synthesised %d bytes for lang=%s, text_len=%d",
            len(audio_bytes), language, len(text),
        )
        return audio_bytes

    except Exception as exc:
        logger.error("TTS synthesis failed for lang=%s: %s", language, exc)
        return b""
