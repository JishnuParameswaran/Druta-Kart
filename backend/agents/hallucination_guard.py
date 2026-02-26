"""
Druta Kart - Hallucination Guard.

Runs on a sampled subset of bot responses (default 50%) to catch fabricated
facts, policy violations, and impossible promises before they reach the customer.

Uses Groq LLM with the v1_hallucination_check prompt.  If the response is
deemed unsafe the corrected version from the LLM is used instead.
"""
from __future__ import annotations

import json
import logging
import random
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_llm():
    from langchain_groq import ChatGroq  # lazy import
    return ChatGroq(
        model=settings.groq_text_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
    )


def _load_check_prompt() -> str:
    try:
        from prompts.prompt_registry import get_prompt
        return get_prompt("hallucination_check")
    except Exception as exc:
        logger.warning("Could not load hallucination_check prompt: %s", exc)
        return (
            'You are a factual checker. Review the response and reply with JSON: '
            '{"safe": true, "issues": [], "corrected_response": null}'
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_response(
    response: str,
    session_id: str = "",
    user_id: str = "",
) -> dict:
    """Validate *response* for hallucinations using Groq LLM.

    This function is always called but applies sampling internally —
    if the random draw misses, the original response is returned unchanged.

    Args:
        response:   The bot-generated response text to validate.
        session_id: For logging only.
        user_id:    For logging only.

    Returns:
        Dict with keys:
            response (str)              — safe or corrected response
            hallucination_flagged (bool)— True if issues were found
            issues (list[str])          — list of detected issues
    """
    if not response or not response.strip():
        return {"response": response, "hallucination_flagged": False, "issues": []}

    # Sampling gate — skip most responses to save LLM cost
    if random.random() > settings.hallucination_check_sampling_rate:
        return {"response": response, "hallucination_flagged": False, "issues": []}

    try:
        prompt_template = _load_check_prompt()
        filled_prompt = prompt_template.replace("{response}", response)

        llm = _get_llm()
        from langchain_core.messages import HumanMessage
        result = llm.invoke([HumanMessage(content=filled_prompt)])
        raw = result.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        safe: bool = data.get("safe", True)
        issues: list = data.get("issues", [])
        corrected: Optional[str] = data.get("corrected_response")

        if not safe and corrected:
            logger.warning(
                "Hallucination detected session=%s user=%s issues=%s",
                session_id, user_id, issues,
            )
            return {
                "response": corrected,
                "hallucination_flagged": True,
                "issues": issues,
            }

        return {"response": response, "hallucination_flagged": False, "issues": []}

    except json.JSONDecodeError as exc:
        logger.error("Hallucination guard: JSON parse error: %s", exc)
        return {"response": response, "hallucination_flagged": False, "issues": []}
    except Exception as exc:
        logger.error("Hallucination guard failed for session=%s: %s", session_id, exc)
        # On failure, pass the response through unchanged — never block the user
        return {"response": response, "hallucination_flagged": False, "issues": []}
