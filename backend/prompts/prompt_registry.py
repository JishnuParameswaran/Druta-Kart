"""
Druta Kart - Prompt Registry

Loads prompt text files by version key.  The active version is controlled
by the PROMPT_VERSION environment variable (default: "v1").

Usage:
    from prompts.prompt_registry import get_prompt
    system = get_prompt("system_prompt")          # uses active version
    hcheck = get_prompt("hallucination_check")    # uses active version
"""
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent

# Map logical name → filename suffix (without version prefix)
_PROMPT_FILES = {
    "system_prompt": "system_prompt.txt",
    "hallucination_check": "hallucination_check.txt",
}


def _active_version() -> str:
    return os.getenv("PROMPT_VERSION", "v1")


@lru_cache(maxsize=None)
def _load_prompt(version: str, name: str) -> str:
    """Load and cache a prompt file.  Raises FileNotFoundError if missing."""
    suffix = _PROMPT_FILES.get(name)
    if suffix is None:
        raise KeyError(f"Unknown prompt name: {name!r}. Available: {list(_PROMPT_FILES)}")

    path = _PROMPTS_DIR / f"{version}_{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    logger.debug("Loaded prompt %s/%s from %s", version, name, path)
    return text


def get_prompt(name: str, version: str | None = None) -> str:
    """Return the text of a prompt by logical name.

    Args:
        name:    Logical prompt name ("system_prompt" or "hallucination_check").
        version: Override the active version (e.g. "v2").  Defaults to
                 the PROMPT_VERSION env var (default "v1").

    Returns:
        Prompt text string.

    Raises:
        KeyError: Unknown prompt name.
        FileNotFoundError: Prompt file for the requested version is missing.
    """
    v = version or _active_version()
    return _load_prompt(v, name)


def clear_cache() -> None:
    """Invalidate the prompt cache (useful in tests or after hot-reload)."""
    _load_prompt.cache_clear()
