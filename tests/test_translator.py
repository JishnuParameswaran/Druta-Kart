"""
Tests for backend/nlp/translator.py

All tests mock requests.post — no real network calls are made.
"""
import os
from unittest.mock import MagicMock, patch

import pytest
import requests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(translated_text: str, status_code: int = 200):
    """Build a fake requests.Response object."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {"translated_text": translated_text}
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_resp
        )
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


# ---------------------------------------------------------------------------
# translate_to_english
# ---------------------------------------------------------------------------

class TestTranslateToEnglish:
    @pytest.fixture(autouse=True)
    def set_api_key(self, monkeypatch):
        monkeypatch.setenv("SARVAM_API_KEY", "test-key-123")

    def test_english_source_returns_original_no_api_call(self):
        from nlp.translator import translate_to_english
        with patch("nlp.translator.requests.post") as mock_post:
            result = translate_to_english("Hello there", "en-IN")
        mock_post.assert_not_called()
        assert result == "Hello there"

    def test_bare_en_code_returns_original(self):
        from nlp.translator import translate_to_english
        with patch("nlp.translator.requests.post") as mock_post:
            result = translate_to_english("Hello", "en")
        mock_post.assert_not_called()
        assert result == "Hello"

    def test_en_us_returns_original(self):
        from nlp.translator import translate_to_english
        with patch("nlp.translator.requests.post") as mock_post:
            result = translate_to_english("Hello", "en-US")
        mock_post.assert_not_called()
        assert result == "Hello"

    def test_hindi_source_calls_api_and_returns_translation(self):
        from nlp.translator import translate_to_english
        mock_resp = _mock_response("My order has not arrived yet")
        with patch("nlp.translator.requests.post", return_value=mock_resp):
            result = translate_to_english("मेरा ऑर्डर नहीं आया", "hi-IN")
        assert result == "My order has not arrived yet"

    def test_tamil_source_calls_api(self):
        from nlp.translator import translate_to_english
        mock_resp = _mock_response("Where is my order")
        with patch("nlp.translator.requests.post", return_value=mock_resp):
            result = translate_to_english("என் ஆர்டர் எங்கே", "ta-IN")
        assert result == "Where is my order"

    def test_correct_payload_sent_to_api(self):
        from nlp.translator import translate_to_english, SARVAM_TRANSLATE_URL
        mock_resp = _mock_response("translated")
        with patch("nlp.translator.requests.post", return_value=mock_resp) as mock_post:
            translate_to_english("मेरा ऑर्डर", "hi-IN")

        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == SARVAM_TRANSLATE_URL
        payload = call_kwargs[1]["json"]
        assert payload["source_language_code"] == "hi-IN"
        assert payload["target_language_code"] == "en-IN"
        assert payload["input"] == "मेरा ऑर्डर"

    def test_api_key_sent_in_header(self):
        from nlp.translator import translate_to_english
        mock_resp = _mock_response("translated")
        with patch("nlp.translator.requests.post", return_value=mock_resp) as mock_post:
            translate_to_english("मेरा ऑर्डर", "hi-IN")

        headers = mock_post.call_args[1]["headers"]
        assert headers["api-subscription-key"] == "test-key-123"

    def test_timeout_set_to_10_seconds(self):
        from nlp.translator import translate_to_english
        mock_resp = _mock_response("translated")
        with patch("nlp.translator.requests.post", return_value=mock_resp) as mock_post:
            translate_to_english("मेरा ऑर्डर", "hi-IN")

        assert mock_post.call_args[1]["timeout"] == 10

    def test_timeout_exception_returns_original(self):
        from nlp.translator import translate_to_english
        with patch("nlp.translator.requests.post",
                   side_effect=requests.exceptions.Timeout()):
            result = translate_to_english("मेरा ऑर्डर नहीं आया", "hi-IN")
        assert result == "मेरा ऑर्डर नहीं आया"

    def test_http_error_returns_original(self):
        from nlp.translator import translate_to_english
        mock_resp = _mock_response("", status_code=500)
        with patch("nlp.translator.requests.post", return_value=mock_resp):
            result = translate_to_english("मेरा ऑर्डर", "hi-IN")
        assert result == "मेरा ऑर्डर"

    def test_empty_api_response_returns_original(self):
        from nlp.translator import translate_to_english
        mock_resp = _mock_response("")          # empty translated_text
        with patch("nlp.translator.requests.post", return_value=mock_resp):
            result = translate_to_english("मेरा ऑर्डर", "hi-IN")
        assert result == "मेरा ऑर्डर"

    def test_empty_input_returns_empty_no_api_call(self):
        from nlp.translator import translate_to_english
        with patch("nlp.translator.requests.post") as mock_post:
            result = translate_to_english("", "hi-IN")
        mock_post.assert_not_called()
        assert result == ""

    def test_whitespace_input_returns_whitespace_no_api_call(self):
        from nlp.translator import translate_to_english
        with patch("nlp.translator.requests.post") as mock_post:
            result = translate_to_english("   ", "hi-IN")
        mock_post.assert_not_called()
        assert result == "   "

    def test_missing_api_key_returns_original(self, monkeypatch):
        from nlp import translator as mod
        monkeypatch.delenv("SARVAM_API_KEY", raising=False)
        # Reset cached env read
        with patch.object(mod, "_get_api_key", return_value=None):
            result = mod.translate_to_english("मेरा ऑर्डर", "hi-IN")
        assert result == "मेरा ऑर्डर"

    def test_unexpected_exception_returns_original(self):
        from nlp.translator import translate_to_english
        with patch("nlp.translator.requests.post",
                   side_effect=ConnectionError("network down")):
            result = translate_to_english("मेरा ऑर्डर", "hi-IN")
        assert result == "मेरा ऑर्डर"


# ---------------------------------------------------------------------------
# translate_to_language
# ---------------------------------------------------------------------------

class TestTranslateToLanguage:
    @pytest.fixture(autouse=True)
    def set_api_key(self, monkeypatch):
        monkeypatch.setenv("SARVAM_API_KEY", "test-key-123")

    def test_english_target_returns_original_no_api_call(self):
        from nlp.translator import translate_to_language
        with patch("nlp.translator.requests.post") as mock_post:
            result = translate_to_language("Your order is on the way.", "en-IN")
        mock_post.assert_not_called()
        assert result == "Your order is on the way."

    def test_bare_en_target_returns_original(self):
        from nlp.translator import translate_to_language
        with patch("nlp.translator.requests.post") as mock_post:
            result = translate_to_language("Hello", "en")
        mock_post.assert_not_called()
        assert result == "Hello"

    def test_hindi_target_calls_api_and_returns_translation(self):
        from nlp.translator import translate_to_language
        mock_resp = _mock_response("आपका ऑर्डर रास्ते में है")
        with patch("nlp.translator.requests.post", return_value=mock_resp):
            result = translate_to_language("Your order is on the way.", "hi-IN")
        assert result == "आपका ऑर्डर रास्ते में है"

    def test_malayalam_target_calls_api(self):
        from nlp.translator import translate_to_language
        mock_resp = _mock_response("നിങ്ങളുടെ ഓർഡർ വഴിയിലാണ്")
        with patch("nlp.translator.requests.post", return_value=mock_resp):
            result = translate_to_language("Your order is on the way.", "ml-IN")
        assert result == "നിങ്ങളുടെ ഓർഡർ വഴിയിലാണ്"

    def test_correct_payload_sent_to_api(self):
        from nlp.translator import translate_to_language, SARVAM_TRANSLATE_URL
        mock_resp = _mock_response("translated")
        with patch("nlp.translator.requests.post", return_value=mock_resp) as mock_post:
            translate_to_language("Your order is ready.", "kn-IN")

        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == SARVAM_TRANSLATE_URL
        payload = call_kwargs[1]["json"]
        assert payload["source_language_code"] == "en-IN"
        assert payload["target_language_code"] == "kn-IN"
        assert payload["input"] == "Your order is ready."

    def test_timeout_exception_returns_original(self):
        from nlp.translator import translate_to_language
        with patch("nlp.translator.requests.post",
                   side_effect=requests.exceptions.Timeout()):
            result = translate_to_language("Your order is ready.", "hi-IN")
        assert result == "Your order is ready."

    def test_http_error_returns_original(self):
        from nlp.translator import translate_to_language
        mock_resp = _mock_response("", status_code=401)
        with patch("nlp.translator.requests.post", return_value=mock_resp):
            result = translate_to_language("Your order is ready.", "hi-IN")
        assert result == "Your order is ready."

    def test_empty_input_returns_empty_no_api_call(self):
        from nlp.translator import translate_to_language
        with patch("nlp.translator.requests.post") as mock_post:
            result = translate_to_language("", "hi-IN")
        mock_post.assert_not_called()
        assert result == ""

    def test_empty_api_response_returns_original(self):
        from nlp.translator import translate_to_language
        mock_resp = _mock_response("")
        with patch("nlp.translator.requests.post", return_value=mock_resp):
            result = translate_to_language("Hello", "hi-IN")
        assert result == "Hello"

    def test_unexpected_exception_returns_original(self):
        from nlp.translator import translate_to_language
        with patch("nlp.translator.requests.post",
                   side_effect=ValueError("bad json")):
            result = translate_to_language("Hello", "ta-IN")
        assert result == "Hello"

    def test_all_indian_language_codes_accepted(self):
        """Verify each supported non-English code triggers an API call."""
        from nlp.translator import translate_to_language
        indian_codes = ["hi-IN", "ml-IN", "ta-IN", "kn-IN", "mr-IN"]
        for code in indian_codes:
            mock_resp = _mock_response(f"translated to {code}")
            with patch("nlp.translator.requests.post", return_value=mock_resp) as mock_post:
                result = translate_to_language("Test message", code)
            mock_post.assert_called_once()
            assert result == f"translated to {code}"
