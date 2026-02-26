"""
Druta Kart - Fraud Escalation Agent.

Triggered when image_validation_agent confirms a suspicious or AI-generated
image.  Logs a fraud flag to the database and returns a polite response that
never directly accuses the customer.
"""
from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)

# Polite, non-accusatory response template
_REVIEW_MESSAGE = (
    "Thank you for reaching out. We want to make sure every case is handled "
    "fairly and thoroughly. Our quality assurance team will review the details "
    "of your complaint and get back to you within 2 hours. We appreciate your "
    "patience and will do our best to resolve this quickly."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_fraud_flag(
    user_id: str,
    session_id: str,
    reason: str,
    image_path: str,
) -> None:
    """Persist a fraud flag to the database (fire-and-forget)."""
    try:
        from db.supabase_client import get_client
        client = get_client()
        client.table("fraud_flags").insert({
            "flag_id": str(uuid4()),
            "user_id": user_id,
            "session_id": session_id,
            "reason": reason,
            "image_path": image_path,
            "status": "pending_review",
            "created_at": datetime.utcnow().isoformat(),
        }).execute()
        logger.info(
            "Fraud flag logged: user=%s session=%s reason=%s",
            user_id, session_id, reason,
        )
    except Exception as exc:
        logger.error("Failed to log fraud flag for user=%s: %s", user_id, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(state: dict) -> dict:
    """Handle a confirmed suspicious image submission.

    Logs the fraud event and returns a polite escalation message.
    Never accuses the customer directly.

    Args:
        state: AgentState dict with at least user_id, session_id,
               image_path, image_validation_result.

    Returns:
        Partial AgentState update dict.
    """
    user_id = state.get("user_id", "unknown")
    session_id = state.get("session_id", "unknown")
    image_path = state.get("image_path", "")
    validation_result = state.get("image_validation_result", "suspicious")

    reason = f"Image classified as: {validation_result}"
    _log_fraud_flag(user_id, session_id, reason, image_path)

    logger.warning(
        "Fraud escalation triggered: user=%s session=%s result=%s",
        user_id, session_id, validation_result,
    )

    return {
        "response": _REVIEW_MESSAGE,
        "resolved": False,
        "fraud_flagged": True,
        "tools_called": state.get("tools_called", []) + ["fraud_escalation_agent"],
    }
