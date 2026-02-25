"""
Language detection using lingua-language-detector.

Detects Indian languages (Hindi, Malayalam, Tamil, Kannada, Marathi) and English.
Handles code-mixed text (Hinglish = Hindi+English, Kanglish = Kannada+English)
by inspecting script character ratios in addition to lingua's confidence scores.

Returns ISO-style language codes: hi-IN, ml-IN, ta-IN, kn-IN, mr-IN, en-IN
"""
from __future__ import annotations

import logging
import unicodedata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language code mapping
# ---------------------------------------------------------------------------

LINGUA_TO_ISO: dict[str, str] = {
    "HINDI": "hi-IN",
    "MALAYALAM": "ml-IN",
    "TAMIL": "ta-IN",
    "KANNADA": "kn-IN",
    "MARATHI": "mr-IN",
    "ENGLISH": "en-IN",
}

ISO_TO_LINGUA: dict[str, str] = {v: k for k, v in LINGUA_TO_ISO.items()}

# Unicode script block ranges used for mixed-script detection
DEVANAGARI_RANGE = (0x0900, 0x097F)   # Hindi, Marathi
MALAYALAM_RANGE = (0x0D00, 0x0D7F)
TAMIL_RANGE = (0x0B80, 0x0BFF)
KANNADA_RANGE = (0x0C80, 0x0CFF)

# Minimum fraction of native-script chars to consider text "native" rather than romanised
SCRIPT_THRESHOLD = 0.10

# lingua confidence threshold below which we fall back to script analysis
CONFIDENCE_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Lazy detector initialisation
# ---------------------------------------------------------------------------

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        try:
            from lingua import Language, LanguageDetectorBuilder  # noqa: PLC0415

            _detector = (
                LanguageDetectorBuilder.from_languages(
                    Language.ENGLISH,
                    Language.HINDI,
                    Language.MALAYALAM,
                    Language.TAMIL,
                    Language.KANNADA,
                    Language.MARATHI,
                )
                .with_preloaded_language_models()
                .build()
            )
            logger.info("Lingua language detector initialised for 6 languages.")
        except Exception as exc:
            logger.error("Failed to initialise lingua detector: %s", exc)
            raise
    return _detector


# ---------------------------------------------------------------------------
# Script-ratio helpers
# ---------------------------------------------------------------------------

def _script_ratio(text: str, lo: int, hi: int) -> float:
    """Fraction of alphabetic characters that fall in the Unicode range [lo, hi]."""
    alpha_chars = [c for c in text if unicodedata.category(c).startswith("L")]
    if not alpha_chars:
        return 0.0
    in_range = sum(1 for c in alpha_chars if lo <= ord(c) <= hi)
    return in_range / len(alpha_chars)


def _dominant_native_script(text: str) -> str | None:
    """
    Return the ISO code of the dominant native script if any script exceeds
    SCRIPT_THRESHOLD, else None.
    """
    checks = [
        ("hi-IN", *DEVANAGARI_RANGE),
        ("ml-IN", *MALAYALAM_RANGE),
        ("ta-IN", *TAMIL_RANGE),
        ("kn-IN", *KANNADA_RANGE),
    ]
    best_lang, best_ratio = None, 0.0
    for lang_code, lo, hi in checks:
        ratio = _script_ratio(text, lo, hi)
        if ratio > best_ratio:
            best_ratio = ratio
            best_lang = lang_code

    # Marathi uses Devanagari too — lingua is relied on to distinguish it from Hindi
    return best_lang if best_ratio >= SCRIPT_THRESHOLD else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """
    Detect the language of *text* and return an ISO code.

    Code-mixed handling (Hinglish / Kanglish):
    - If the text contains significant native-script characters the dominant
      script language is preferred.
    - If the text is fully romanised (no native script) and lingua confidence
      is low, we fall back to "en-IN" (Hinglish / Kanglish typed in Latin
      script is treated as English for the LLM pipeline).

    Returns one of: hi-IN, ml-IN, ta-IN, kn-IN, mr-IN, en-IN
    Default: en-IN
    """
    if not text or not text.strip():
        return "en-IN"

    # 1. Script-ratio check for native scripts
    script_lang = _dominant_native_script(text)

    try:
        detector = _get_detector()

        # lingua returns the most likely language
        result = detector.detect_language_of(text)
        if result is None:
            return script_lang or "en-IN"

        lingua_label = result.name.upper()  # e.g. "HINDI"
        iso_code = LINGUA_TO_ISO.get(lingua_label, "en-IN")

        # 2. Get confidence
        confidence_values = detector.compute_language_confidence_values(text)
        confidence_map = {cv.language.name.upper(): cv.value for cv in confidence_values}
        top_confidence = confidence_map.get(lingua_label, 0.0)

        # 3. Resolve code-mixed text
        if script_lang and script_lang != iso_code:
            # Native script strongly suggests a language — trust it over lingua
            if top_confidence < CONFIDENCE_THRESHOLD:
                logger.debug(
                    "Script override: lingua=%s (%.2f) -> script=%s",
                    iso_code,
                    top_confidence,
                    script_lang,
                )
                return script_lang

        # 4. Low-confidence romanised text → treat as English (Hinglish/Kanglish)
        if top_confidence < CONFIDENCE_THRESHOLD and script_lang is None:
            logger.debug(
                "Low confidence (%.2f) for %s, no native script — defaulting to en-IN",
                top_confidence,
                lingua_label,
            )
            return "en-IN"

        return iso_code

    except Exception as exc:
        logger.warning("Language detection failed: %s", exc)
        return script_lang or "en-IN"


def is_english(text: str) -> bool:
    """Return True if *text* is detected as English (including Hinglish/Kanglish)."""
    return detect_language(text) == "en-IN"
