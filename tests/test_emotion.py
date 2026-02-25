"""
Tests for backend/nlp/emotion_analyzer.py

The HuggingFace model is mocked in all tests — we verify the logic that
wraps the pipeline, not the model weights themselves.
"""
import sys
import importlib
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_mock(dominant_label: str, dominant_score: float = 0.90):
    """
    Return a callable that mimics the HuggingFace text-classification pipeline
    with top_k=None.  All remaining labels share the leftover probability.
    """
    all_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    leftover = round((1.0 - dominant_score) / (len(all_labels) - 1), 4)

    scores = []
    for lbl in all_labels:
        scores.append(
            {"label": lbl, "score": dominant_score if lbl == dominant_label else leftover}
        )

    mock = MagicMock(return_value=[scores])  # pipeline returns list-of-list
    return mock


def _fresh_module():
    """Re-import emotion_analyzer with a clean _pipeline state."""
    import nlp.emotion_analyzer as mod
    mod._pipeline = None          # reset cached pipeline
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnalyzeEmotionEmptyInput:
    def test_empty_string_returns_neutral(self):
        import nlp.emotion_analyzer as mod
        result = mod.analyze_emotion("")
        assert result["label"] == "neutral"
        assert result["score"] == 1.0
        assert result["urgent"] is False

    def test_whitespace_only_returns_neutral(self):
        import nlp.emotion_analyzer as mod
        result = mod.analyze_emotion("   ")
        assert result["label"] == "neutral"
        assert result["urgent"] is False

    def test_none_equivalent_empty(self):
        import nlp.emotion_analyzer as mod
        # Passing an empty string (falsy)
        result = mod.analyze_emotion("")
        assert "all" in result
        assert result["all"][0]["label"] == "neutral"


class TestAnalyzeEmotionWithMockedPipeline:
    @pytest.fixture(autouse=True)
    def reset_pipeline(self):
        """Ensure the module-level pipeline cache is cleared before each test."""
        import nlp.emotion_analyzer as mod
        mod._pipeline = None
        yield
        mod._pipeline = None

    def test_angry_text_returns_anger(self):
        import nlp.emotion_analyzer as mod
        mock_pipe = _make_pipeline_mock("anger", 0.92)
        with patch.object(mod, "_get_pipeline", return_value=mock_pipe):
            result = mod.analyze_emotion("This is absolutely terrible! I am furious!")
        assert result["label"] == "anger"
        assert result["urgent"] is True
        assert result["score"] == pytest.approx(0.92, abs=0.001)

    def test_happy_text_returns_joy(self):
        import nlp.emotion_analyzer as mod
        mock_pipe = _make_pipeline_mock("joy", 0.88)
        with patch.object(mod, "_get_pipeline", return_value=mock_pipe):
            result = mod.analyze_emotion("I love this product, it made my day!")
        assert result["label"] == "joy"
        assert result["urgent"] is False

    def test_sad_text_returns_sadness(self):
        import nlp.emotion_analyzer as mod
        mock_pipe = _make_pipeline_mock("sadness", 0.85)
        with patch.object(mod, "_get_pipeline", return_value=mock_pipe):
            result = mod.analyze_emotion("I am so disappointed, my order never arrived.")
        assert result["label"] == "sadness"
        assert result["urgent"] is False

    def test_fear_is_urgent(self):
        import nlp.emotion_analyzer as mod
        mock_pipe = _make_pipeline_mock("fear", 0.80)
        with patch.object(mod, "_get_pipeline", return_value=mock_pipe):
            result = mod.analyze_emotion("I'm scared I was charged twice!")
        assert result["label"] == "fear"
        assert result["urgent"] is True

    def test_disgust_is_urgent(self):
        import nlp.emotion_analyzer as mod
        mock_pipe = _make_pipeline_mock("disgust", 0.78)
        with patch.object(mod, "_get_pipeline", return_value=mock_pipe):
            result = mod.analyze_emotion("This is disgusting, the food was rotten.")
        assert result["label"] == "disgust"
        assert result["urgent"] is True

    def test_neutral_is_not_urgent(self):
        import nlp.emotion_analyzer as mod
        mock_pipe = _make_pipeline_mock("neutral", 0.70)
        with patch.object(mod, "_get_pipeline", return_value=mock_pipe):
            result = mod.analyze_emotion("Where is my order?")
        assert result["label"] == "neutral"
        assert result["urgent"] is False

    def test_hinglish_input_is_processed(self):
        """Hinglish (romanised Hindi+English) should go through the pipeline unchanged."""
        import nlp.emotion_analyzer as mod
        mock_pipe = _make_pipeline_mock("anger", 0.83)
        with patch.object(mod, "_get_pipeline", return_value=mock_pipe):
            result = mod.analyze_emotion("Yaar mera order abhi tak nahi aaya, bahut gussa aa raha hai!")
        assert result["label"] == "anger"
        assert result["urgent"] is True

    def test_kanglish_input_is_processed(self):
        """Kanglish (romanised Kannada+English) should go through the pipeline unchanged."""
        import nlp.emotion_analyzer as mod
        mock_pipe = _make_pipeline_mock("sadness", 0.76)
        with patch.object(mod, "_get_pipeline", return_value=mock_pipe):
            result = mod.analyze_emotion("Nanna order barthilla, tumba sad aagide naanu")
        assert result["label"] == "sadness"
        assert result["urgent"] is False

    def test_result_contains_all_keys(self):
        import nlp.emotion_analyzer as mod
        mock_pipe = _make_pipeline_mock("joy", 0.75)
        with patch.object(mod, "_get_pipeline", return_value=mock_pipe):
            result = mod.analyze_emotion("Great!")
        assert set(result.keys()) >= {"label", "score", "all", "urgent"}

    def test_all_field_is_sorted_descending(self):
        import nlp.emotion_analyzer as mod
        mock_pipe = _make_pipeline_mock("anger", 0.91)
        with patch.object(mod, "_get_pipeline", return_value=mock_pipe):
            result = mod.analyze_emotion("I hate this!")
        scores = [item["score"] for item in result["all"]]
        assert scores == sorted(scores, reverse=True)

    def test_long_text_is_truncated_safely(self):
        """Text longer than 512 chars should not raise."""
        import nlp.emotion_analyzer as mod
        mock_pipe = _make_pipeline_mock("neutral", 0.60)
        long_text = "My order is missing. " * 50          # > 512 chars
        with patch.object(mod, "_get_pipeline", return_value=mock_pipe):
            result = mod.analyze_emotion(long_text)
        assert "label" in result

    def test_pipeline_failure_returns_neutral_with_error(self):
        """If the pipeline raises, analyze_emotion must degrade gracefully."""
        import nlp.emotion_analyzer as mod

        def _exploding_pipe(_text):
            raise RuntimeError("model crashed")

        with patch.object(mod, "_get_pipeline", return_value=_exploding_pipe):
            result = mod.analyze_emotion("test text")

        assert result["label"] == "neutral"
        assert result["urgent"] is False
        assert "error" in result


class TestIsUrgent:
    @pytest.fixture(autouse=True)
    def reset_pipeline(self):
        import nlp.emotion_analyzer as mod
        mod._pipeline = None
        yield
        mod._pipeline = None

    def test_angry_text_is_urgent(self):
        import nlp.emotion_analyzer as mod
        mock_pipe = _make_pipeline_mock("anger", 0.93)
        with patch.object(mod, "_get_pipeline", return_value=mock_pipe):
            assert mod.is_urgent("Refund my money NOW!") is True

    def test_joy_text_is_not_urgent(self):
        import nlp.emotion_analyzer as mod
        mock_pipe = _make_pipeline_mock("joy", 0.88)
        with patch.object(mod, "_get_pipeline", return_value=mock_pipe):
            assert mod.is_urgent("Thank you so much!") is False

    def test_empty_text_is_not_urgent(self):
        import nlp.emotion_analyzer as mod
        assert mod.is_urgent("") is False
