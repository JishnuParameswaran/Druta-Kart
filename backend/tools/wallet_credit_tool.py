"""
Druta Kart - Wallet credit tool.

LangChain tool that credits the customer's Druta Kart wallet.

SAFETY CAP: The credited amount is hard-clamped to settings.max_wallet_credit_inr
(default ₹200) per transaction.  This cap is enforced in Python and cannot
be bypassed by the LLM.
"""
import logging
from datetime import datetime
from uuid import uuid4

from langchain_core.tools import tool

from config import settings
from db.supabase_client import get_client

logger = logging.getLogger(__name__)


@tool
def wallet_credit_tool(user_id: str, amount_inr: float, reason: str) -> dict:
    """Credit an amount to the customer's Druta Kart wallet as a goodwill gesture.

    The amount is automatically capped at the configured safety limit
    (₹200 by default) regardless of the value passed in.

    Args:
        user_id:    The customer's user ID.
        amount_inr: Desired credit amount in Indian Rupees (will be capped).
        reason:     Plain-text reason for the credit.

    Returns:
        Dict with keys: user_id, credited_amount, cap_applied (bool),
        reason, message.  On failure: status="failed" and 'error'.
    """
    try:
        # Enforce safety cap — LLM cannot override this
        capped_amount = round(
            min(float(settings.max_wallet_credit_inr), max(0.0, float(amount_inr))), 2
        )
        cap_applied = capped_amount < float(amount_inr)

        if cap_applied:
            logger.info(
                "wallet_credit_tool: amount %.2f capped to %.2f for user=%s",
                amount_inr, capped_amount, user_id,
            )

        client = get_client()
        client.table("wallet_transactions").insert({
            "transaction_id": str(uuid4()),
            "user_id": user_id,
            "amount_inr": capped_amount,
            "type": "credit",
            "reason": reason,
            "created_at": datetime.utcnow().isoformat(),
        }).execute()

        return {
            "user_id": user_id,
            "credited_amount": capped_amount,
            "cap_applied": cap_applied,
            "reason": reason,
            "message": f"₹{int(capped_amount)} has been added to your Druta Kart wallet.",
        }
    except Exception as exc:
        logger.error("wallet_credit_tool failed for user=%s: %s", user_id, exc)
        return {"user_id": user_id, "status": "failed", "error": str(exc)}
