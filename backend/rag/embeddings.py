"""
Druta Kart - Sentence-Transformer Embedding Function.

Provides a ChromaDB-compatible EmbeddingFunction that wraps
sentence-transformers. Falls back to a deterministic hash-based dummy
when the library is unavailable (unit-test / cold-start safety).

Default model: all-MiniLM-L6-v2 (384-dim, CPU-friendly, ~80 MB).
"""
from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_model_instance = None


def _get_model():
    """Return a cached SentenceTransformer model, or None if unavailable."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _model_instance = SentenceTransformer(_EMBEDDING_MODEL)
        logger.info("Loaded sentence-transformer model: %s", _EMBEDDING_MODEL)
        return _model_instance
    except Exception as exc:
        logger.warning("sentence-transformers unavailable (%s); using dummy.", exc)
        return None


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts and return raw 384-dim float vectors."""
    model = _get_model()
    if model is not None:
        return model.encode(texts, show_progress_bar=False).tolist()
    return _DummyEmbeddingFunction()(texts)


def get_embedding_function():
    """Return a ChromaDB-compatible embedding function.

    Tries sentence-transformers first; falls back to a dummy implementation
    that allows ChromaDB to operate without ML inference (useful in tests
    and environments where torch/sentence-transformers is not installed).

    Returns:
        Object with ``__call__(input: List[str]) -> List[List[float]]``
        matching ChromaDB's EmbeddingFunction protocol.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore  # noqa: F401
        return _SentenceTransformerEmbeddingFunction(model_name=_EMBEDDING_MODEL)
    except ImportError:
        logger.warning(
            "sentence-transformers not available; using dummy embedding function."
        )
        return _DummyEmbeddingFunction()


class _SentenceTransformerEmbeddingFunction:
    """ChromaDB EmbeddingFunction backed by sentence-transformers."""

    def __init__(self, model_name: str = _EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._model = SentenceTransformer(model_name)
        logger.info("Loaded sentence-transformer model: %s", model_name)

    def name(self) -> str:
        return "sentence-transformer"

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        embeddings = self._model.encode(input, show_progress_bar=False)
        return embeddings.tolist()


class _DummyEmbeddingFunction:
    """Fallback embedding function for environments without sentence-transformers.

    Produces a deterministic 384-dim vector via character-level hashing so
    ChromaDB can store and retrieve documents without any ML inference.
    Cosine similarity over these vectors is meaningless; the vector_store
    keyword-search fallback is used for actual retrieval in that case.
    """

    _DIM = 384

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        result = []
        for text in input:
            vec = [0.0] * self._DIM
            for i, ch in enumerate(text):
                vec[i % self._DIM] += ord(ch) / 128.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            result.append([v / norm for v in vec])
        return result
