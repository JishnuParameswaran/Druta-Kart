"""
Druta Kart - Replacement tool.

LangChain tool that raises a replacement request for a damaged, wrong, or
missing item and writes it to Supabase.
"""
import logging
from datetime import datetime, timedelta
from uuid import uuid4

from langchain_core.tools import tool

from db.supabase_client import get_client

logger = logging.getLogger(__name__)

_REPLACEMENT_HOURS = 24  # standard same/next-day replacement SLA


@tool
def replacement_tool(order_id: str, user_id: str, product_name: str, reason: str) -> dict:
    """Initiate a replacement for a damaged, wrong, or missing item.

    Use this instead of refund_tool when the customer wants the item
    re-delivered rather than a monetary refund.

    Args:
        order_id:     The original order ID.
        user_id:      The customer's user ID.
        product_name: Name of the item to be replaced.
        reason:       Reason for replacement (e.g. "item arrived damaged").

    Returns:
        Dict with keys: replacement_id, order_id, status, product_name,
        expected_delivery, message.  On failure: status="failed" and 'error'.
    """
    try:
        replacement_id = str(uuid4())
        now = datetime.utcnow()
        expected_delivery = (now + timedelta(hours=_REPLACEMENT_HOURS)).strftime("%Y-%m-%d")

        client = get_client()
        client.table("replacement_requests").insert({
            "replacement_id": replacement_id,
            "order_id": order_id,
            "user_id": user_id,
            "product_name": product_name,
            "reason": reason,
            "status": "approved",
            "expected_delivery": expected_delivery,
            "created_at": now.isoformat(),
        }).execute()

        return {
            "replacement_id": replacement_id,
            "order_id": order_id,
            "status": "approved",
            "product_name": product_name,
            "expected_delivery": expected_delivery,
            "message": (
                f"Replacement for '{product_name}' has been approved. "
                f"Expected delivery by {expected_delivery}."
            ),
        }
    except Exception as exc:
        logger.error("replacement_tool failed for order=%s: %s", order_id, exc)
        return {"order_id": order_id, "status": "failed", "error": str(exc)}
