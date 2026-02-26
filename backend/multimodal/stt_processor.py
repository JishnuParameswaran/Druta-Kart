"""
Druta Kart - Speech-to-Text Processor.

Uses Groq Whisper (whisper-large-v3) to transcribe audio files.
Supports any format accepted by Whisper: mp3, mp4, mpeg, mpga, m4a, wav, webm.
"""
from __future__ import annotations

import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file to text using Groq Whisper.

    Args:
        audio_path: Local path to the audio file.

    Returns:
        Transcribed text string, or empty string on failure.
    """
    path = Path(audio_path)
    if not path.exists():
        logger.error("STT: audio file not found: %s", audio_path)
        return ""

    try:
        from groq import Groq  # lazy import

        client = Groq(api_key=settings.groq_api_key)
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(path.name, f),
                model=settings.groq_whisper_model,
                response_format="text",
            )

        text = transcription if isinstance(transcription, str) else transcription.text
        text = (text or "").strip()
        logger.info("STT: transcribed %d chars from %s", len(text), path.name)
        return text

    except Exception as exc:
        logger.error("STT transcription failed for %s: %s", audio_path, exc)
        return ""
