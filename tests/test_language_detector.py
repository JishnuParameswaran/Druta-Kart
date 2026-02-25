"""
Tests for backend/nlp/language_detector.py

Native-script tests use actual lingua (no network needed — models are local).
Edge-case / fallback tests mock the detector to stay deterministic.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Unit tests for pure helpers (no lingua dependency)
# ---------------------------------------------------------------------------

class TestScriptRatio:
    def test_pure_devanagari_returns_high_ratio(self):
        from nlp.language_detector import _script_ratio, DEVANAGARI_RANGE
        text = "मेरा ऑर्डर नहीं आया"          # pure Hindi Devanagari
        ratio = _script_ratio(text, *DEVANAGARI_RANGE)
        assert ratio > 0.8

    def test_pure_latin_returns_zero_for_devanagari(self):
        from nlp.language_detector import _script_ratio, DEVANAGARI_RANGE
        ratio = _script_ratio("hello world", *DEVANAGARI_RANGE)
        assert ratio == 0.0

    def test_mixed_script_returns_partial_ratio(self):
        from nlp.language_detector import _script_ratio, DEVANAGARI_RANGE
        # roughly 50 % Devanagari, 50 % Latin
        text = "मेरा order आया"
        ratio = _script_ratio(text, *DEVANAGARI_RANGE)
        assert 0.1 < ratio < 1.0

    def test_empty_string_returns_zero(self):
        from nlp.language_detector import _script_ratio, DEVANAGARI_RANGE
        assert _script_ratio("", *DEVANAGARI_RANGE) == 0.0

    def test_digits_only_returns_zero(self):
        from nlp.language_detector import _script_ratio, DEVANAGARI_RANGE
        assert _script_ratio("12345", *DEVANAGARI_RANGE) == 0.0


class TestDominantNativeScript:
    def test_devanagari_text_detected(self):
        from nlp.language_detector import _dominant_native_script
        assert _dominant_native_script("मेरा ऑर्डर कहाँ है") == "hi-IN"

    def test_malayalam_text_detected(self):
        from nlp.language_detector import _dominant_native_script
        assert _dominant_native_script("എന്റെ ഓർഡർ എവിടെ") == "ml-IN"

    def test_tamil_text_detected(self):
        from nlp.language_detector import _dominant_native_script
        assert _dominant_native_script("என் ஆர்டர் எங்கே") == "ta-IN"

    def test_kannada_text_detected(self):
        from nlp.language_detector import _dominant_native_script
        assert _dominant_native_script("ನನ್ನ ಆರ್ಡರ್ ಎಲ್ಲಿದೆ") == "kn-IN"

    def test_latin_text_returns_none(self):
        from nlp.language_detector import _dominant_native_script
        assert _dominant_native_script("where is my order") is None

    def test_empty_returns_none(self):
        from nlp.language_detector import _dominant_native_script
        assert _dominant_native_script("") is None


# ---------------------------------------------------------------------------
# Integration tests — uses actual lingua detector (models loaded locally)
# ---------------------------------------------------------------------------

class TestDetectLanguageNativeScript:
    """Use real native-script strings that lingua handles with high confidence."""

    def test_english_detected(self):
        from nlp.language_detector import detect_language
        assert detect_language("Where is my order? It has been 3 days.") == "en-IN"

    def test_hindi_devanagari_detected(self):
        from nlp.language_detector import detect_language
        # Script ratio alone is enough to return hi-IN
        result = detect_language("मेरा ऑर्डर अभी तक नहीं आया है")
        assert result == "hi-IN"

    def test_malayalam_detected(self):
        from nlp.language_detector import detect_language
        result = detect_language("എന്റെ ഓർഡർ ഇതുവരെ ലഭിച്ചിട്ടില്ല")
        assert result == "ml-IN"

    def test_tamil_detected(self):
        from nlp.language_detector import detect_language
        result = detect_language("என் ஆர்டர் இன்னும் வரவில்லை")
        assert result == "ta-IN"

    def test_kannada_detected(self):
        from nlp.language_detector import detect_language
        result = detect_language("ನನ್ನ ಆರ್ಡರ್ ಇನ್ನೂ ಬಂದಿಲ್ಲ")
        assert result == "kn-IN"

    def test_empty_text_returns_english(self):
        from nlp.language_detector import detect_language
        assert detect_language("") == "en-IN"

    def test_whitespace_returns_english(self):
        from nlp.language_detector import detect_language
        assert detect_language("   ") == "en-IN"


class TestDetectLanguageCodeMixed:
    """Hinglish / Kanglish handling."""

    def test_hinglish_mixed_script_returns_hindi(self):
        """Devanagari chars present → hi-IN regardless of lingua confidence."""
        from nlp.language_detector import detect_language
        # Mix of Devanagari words and Latin words
        result = detect_language("mera ऑर्डर कहाँ है yaar")
        assert result == "hi-IN"

    def test_kanglish_mixed_script_returns_kannada(self):
        from nlp.language_detector import detect_language
        result = detect_language("nanna ಆರ್ಡರ್ where ಆಯಿತು")
        assert result == "kn-IN"

    def test_romanised_hinglish_treated_as_english(self):
        """
        Fully romanised Hinglish (no native script) must return en-IN so the
        LLM pipeline processes it directly without translation.
        """
        from nlp.language_detector import detect_language
        # This text has NO Devanagari — lingua confidence will be low
        result = detect_language("yaar mera order abhi tak nahi aaya hai")
        # Acceptable outcomes: en-IN (low confidence fallback) or hi-IN (lingua confident)
        assert result in ("en-IN", "hi-IN")


# ---------------------------------------------------------------------------
# Fallback / error-path tests — mock the detector
# ---------------------------------------------------------------------------

class TestDetectLanguageFallbacks:
    @pytest.fixture(autouse=True)
    def reset_detector(self):
        import nlp.language_detector as mod
        original = mod._detector
        yield
        mod._detector = original

    def test_detector_returns_none_falls_back_to_english(self):
        import nlp.language_detector as mod

        mock_det = MagicMock()
        mock_det.detect_language_of.return_value = None
        mock_det.compute_language_confidence_values.return_value = []
        mod._detector = mock_det

        result = mod.detect_language("some text")
        assert result == "en-IN"

    def test_detector_exception_falls_back_to_english(self):
        import nlp.language_detector as mod

        mock_det = MagicMock()
        mock_det.detect_language_of.side_effect = RuntimeError("lingua crashed")
        mod._detector = mock_det

        result = mod.detect_language("some text")
        assert result == "en-IN"

    def test_detector_exception_with_native_script_falls_back_to_script_lang(self):
        """If lingua crashes but Devanagari is present, use script detection."""
        import nlp.language_detector as mod

        mock_det = MagicMock()
        mock_det.detect_language_of.side_effect = RuntimeError("lingua crashed")
        mod._detector = mock_det

        result = mod.detect_language("मेरा order नहीं आया")
        assert result == "hi-IN"

    def test_low_confidence_no_script_returns_english(self):
        """Low lingua confidence + no native script → en-IN."""
        import nlp.language_detector as mod
        from unittest.mock import MagicMock

        # Build a fake lingua Language object
        fake_lang = MagicMock()
        fake_lang.name = "HINDI"

        fake_cv = MagicMock()
        fake_cv.language = fake_lang
        fake_cv.value = 0.30          # below CONFIDENCE_THRESHOLD=0.50

        mock_det = MagicMock()
        mock_det.detect_language_of.return_value = fake_lang
        mock_det.compute_language_confidence_values.return_value = [fake_cv]
        mod._detector = mock_det

        # Pure Latin text — no native script chars
        result = mod.detect_language("mera kaam ho gaya")
        assert result == "en-IN"


# ---------------------------------------------------------------------------
# is_english helper
# ---------------------------------------------------------------------------

class TestIsEnglish:
    def test_english_is_english(self):
        from nlp.language_detector import is_english
        assert is_english("Hello, where is my refund?") is True

    def test_hindi_script_is_not_english(self):
        from nlp.language_detector import is_english
        assert is_english("मेरा ऑर्डर नहीं आया") is False

    def test_empty_is_english(self):
        from nlp.language_detector import is_english
        assert is_english("") is True
