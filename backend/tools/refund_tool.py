"""
Druta Kart - Refund tool.

LangChain tool that initiates a refund request and logs it to Supabase.
"""
import logging
from datetime import datetime
from uuid import uuid4

from langchain_core.tools import tool

from db.supabase_client import get_client

logger = logging.getLogger(__name__)

_PROCESSING_DAYS = 5  # standard SLA for refund credit


@tool
def refund_tool(order_id: str, user_id: str, amount_inr: float, reason: str) -> dict:
    """Initiate a refund for a customer order.

    Use this when the customer is entitled to a monetary refund (damaged,
    wrong item delivered, payment failure, etc.).  The refund request is
    written to the database and processed asynchronously by the payments team.

    Args:
        order_id:   The order to be refunded.
        user_id:    The customer's user ID.
        amount_inr: Refund amount in Indian Rupees.
        reason:     Plain-text reason for the refund.

    Returns:
        Dict with keys: refund_id, order_id, status, amount_inr,
        expected_days, message.  On failure: status="failed" and 'error'.
    """
    try:
        refund_id = str(uuid4())
        now = datetime.utcnow().isoformat()

        client = get_client()
        client.table("refund_requests").insert({
            "refund_id": refund_id,
            "order_id": order_id,
            "user_id": user_id,
            "amount_inr": round(float(amount_inr), 2),
            "reason": reason,
            "status": "initiated",
            "created_at": now,
        }).execute()

        return {
            "refund_id": refund_id,
            "order_id": order_id,
            "status": "initiated",
            "amount_inr": round(float(amount_inr), 2),
            "expected_days": _PROCESSING_DAYS,
            "message": (
                f"Refund of ₹{amount_inr:.0f} has been initiated successfully. "
                f"Amount will be credited to your original payment method within "
                f"{_PROCESSING_DAYS}–7 business days."
            ),
        }
    except Exception as exc:
        logger.error("refund_tool failed for order=%s: %s", order_id, exc)
        return {"order_id": order_id, "status": "failed", "error": str(exc)}
