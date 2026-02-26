"""
Druta Kart - Image Validation Agent.

Three-layer validation pipeline for complaint images:
  Layer 1 — EXIF metadata analysis (exifread)
  Layer 2 — Groq Vision LLM analysis (llama-4-scout)
  Layer 3 — Classification into one of four outcomes

Outcomes:
  real_damage       → pass to complaint_agent for resolution
  misidentification → educate customer, ask them to re-check
  ai_generated      → request a live camera photo
  suspicious        → request a live camera photo; escalate on refusal
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification labels
# ---------------------------------------------------------------------------
REAL_DAMAGE = "real_damage"
MISIDENTIFICATION = "misidentification"
AI_GENERATED = "ai_generated"
SUSPICIOUS = "suspicious"

_CUSTOMER_MESSAGES = {
    MISIDENTIFICATION: (
        "Thank you for the photo! Looking at the image, it seems the item may "
        "actually be in the correct condition — it can sometimes appear different "
        "due to packaging or lighting. Could you take another closer look? If it "
        "is genuinely damaged, please take another photo with better lighting and "
        "we will process your complaint right away."
    ),
    AI_GENERATED: (
        "Thank you for reaching out. To process your complaint, we need a live "
        "photo taken directly from your device's camera right now. Please take a "
        "fresh photo of the item and share it with us."
    ),
    SUSPICIOUS: (
        "Thank you for reaching out. To process your complaint accurately, we need "
        "a live photo taken directly from your device's camera right now. Please "
        "take a fresh photo of the item and share it with us."
    ),
}

# ---------------------------------------------------------------------------
# Layer 1 — EXIF analysis
# ---------------------------------------------------------------------------

def _analyse_exif(image_path: str) -> dict:
    """Extract key EXIF metadata and flag anomalies."""
    result: dict = {"has_exif": False, "anomalies": [], "software": None, "datetime": None}
    try:
        import exifread  # type: ignore
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, stop_tag="UNDEF", details=False)

        if not tags:
            result["anomalies"].append("no_exif_data")
            return result

        result["has_exif"] = True
        software = str(tags.get("Image Software", "")).strip()
        result["software"] = software or None

        # Flag known AI-image generation software
        ai_keywords = [
            "stable diffusion", "midjourney", "dall-e", "firefly",
            "bing image creator", "openai", "runway", "leonardoai",
        ]
        if software and any(kw in software.lower() for kw in ai_keywords):
            result["anomalies"].append("ai_software_detected")

        # Missing camera make/model is a soft signal
        has_make = "Image Make" in tags or "EXIF LensMake" in tags
        has_model = "Image Model" in tags
        if not has_make and not has_model:
            result["anomalies"].append("missing_camera_metadata")

        result["datetime"] = str(tags.get("Image DateTime", "")).strip() or None

    except FileNotFoundError:
        logger.warning("image_validation: file not found: %s", image_path)
        result["anomalies"].append("file_not_found")
    except Exception as exc:
        logger.warning("EXIF analysis failed for %s: %s", image_path, exc)

    return result


# ---------------------------------------------------------------------------
# Layer 2 — Groq Vision analysis
# ---------------------------------------------------------------------------

_VISION_PROMPT = (
    "You are a forensic image analyst for a quick-commerce platform. "
    "Analyse this image and determine whether it shows genuine product damage.\n\n"
    "Answer with ONLY one of these labels and a one-sentence reason:\n"
    "- REAL_DAMAGE: The image clearly shows a damaged, wrong, or expired product\n"
    "- MISIDENTIFICATION: The product appears intact; customer may have misidentified damage\n"
    "- AI_GENERATED: Signs of AI generation (unnatural textures, artifacts, impossible lighting)\n"
    "- SUSPICIOUS: The image appears manipulated, re-used, or sourced from the internet\n\n"
    "Format: LABEL | reason"
)


def _analyse_with_vision(image_path: str) -> dict:
    """Call Groq Vision API to classify the image."""
    result: dict = {"label": None, "reason": None}
    try:
        path = Path(image_path)
        if not path.exists():
            result["label"] = SUSPICIOUS
            result["reason"] = "Image file not found"
            return result

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
                    {"type": "text", "text": _VISION_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime};base64,{image_data}"
                    }},
                ],
            }],
            max_tokens=100,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        label_part = raw.split("|")[0].strip().upper() if "|" in raw else raw.split()[0].upper()
        reason_part = raw.split("|", 1)[1].strip() if "|" in raw else raw

        label_map = {
            "REAL_DAMAGE": REAL_DAMAGE,
            "MISIDENTIFICATION": MISIDENTIFICATION,
            "AI_GENERATED": AI_GENERATED,
            "SUSPICIOUS": SUSPICIOUS,
        }
        result["label"] = label_map.get(label_part, SUSPICIOUS)
        result["reason"] = reason_part

    except Exception as exc:
        logger.error("Vision analysis failed for %s: %s", image_path, exc)
        # On API failure, treat as real damage to avoid blocking legitimate complaints
        result["label"] = REAL_DAMAGE
        result["reason"] = f"Vision API unavailable — treating as real damage."

    return result


# ---------------------------------------------------------------------------
# Layer 3 — Final classification
# ---------------------------------------------------------------------------

def _classify(exif: dict, vision: dict) -> str:
    """Merge EXIF and vision signals into a final classification."""
    # Strong EXIF signal overrides vision
    if "ai_software_detected" in exif.get("anomalies", []):
        return AI_GENERATED
    return vision.get("label") or SUSPICIOUS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(state: dict) -> dict:
    """Validate a complaint image through all three layers.

    Args:
        state: AgentState dict with image_path set.

    Returns:
        Partial AgentState update.
    """
    image_path: Optional[str] = state.get("image_path")
    user_id = state.get("user_id", "")
    tools_called = list(state.get("tools_called", []))
    tools_called.append("image_validation_agent")

    if not image_path:
        return {
            "image_validation_result": None,
            "response": (
                "Please share a photo of the item so we can assist you better."
            ),
            "tools_called": tools_called,
        }

    exif = _analyse_exif(image_path)
    vision = _analyse_with_vision(image_path)
    classification = _classify(exif, vision)

    logger.info(
        "Image validation: user=%s result=%s exif_anomalies=%s vision=%s",
        user_id, classification, exif.get("anomalies"), vision.get("label"),
    )

    update: dict = {
        "image_validation_result": classification,
        "tools_called": tools_called,
    }

    if classification == REAL_DAMAGE:
        update["resolved"] = False  # complaint_agent continues
    else:
        update["response"] = _CUSTOMER_MESSAGES[classification]
        update["resolved"] = False

    return update
