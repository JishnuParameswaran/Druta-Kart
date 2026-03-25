"""
Druta Kart - Standalone Image Analyzer.

Provides a single `analyze_image()` function that sends an image to
Groq Vision (llama-4-scout) with a caller-supplied prompt and returns
the raw text response.

The complaint-specific 3-layer validation pipeline lives in
agents/image_validation_agent.py; this module is a lower-level utility.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = (
    "Describe what you see in this image in detail. "
    "Focus on the condition, packaging, and any visible damage or defects."
)


def analyze_image(image_path: str, prompt: str = _DEFAULT_PROMPT) -> str:
    """Analyse an image with Groq Vision and return the model's text response.

    Args:
        image_path: Local path to the image file (JPEG, PNG, WEBP, GIF).
        prompt:     Instruction sent alongside the image.

    Returns:
        Model response string, or empty string on failure.
    """
    path = Path(image_path)
    if not path.exists():
        logger.error("image_analyzer: file not found: %s", image_path)
        return ""

    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = path.suffix.lower().lstrip(".")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

        from groq import Groq  # lazy import

        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.groq_vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_data}"},
                    },
                ],
            }],
            max_tokens=512,
            temperature=0.1,
        )

        text = response.choices[0].message.content.strip()
        logger.info(
            "image_analyzer: %s → %d chars response", path.name, len(text)
        )
        return text

    except Exception as exc:
        logger.error("image_analyzer: vision API failed for %s: %s", image_path, exc)
        return ""
