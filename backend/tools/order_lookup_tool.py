"""
Druta Kart - Order lookup tool.

LangChain tool that fetches current order status and details from Supabase.
"""
import logging

from langchain_core.tools import tool

from db.supabase_client import get_client

logger = logging.getLogger(__name__)


@tool
def order_lookup_tool(order_id: str, user_id: str) -> dict:
    """Fetch current status and full details for a customer order.

    Use this whenever the customer asks "where is my order", "order status",
    or any question about delivery timing for a specific order.

    Args:
        order_id: The unique order identifier (e.g. "ORD-20240112-8821").
        user_id:  The customer's user ID.

    Returns:
        Dict with keys: found (bool), order_id, status, items, amount_inr,
        placed_at, estimated_delivery, delivery_partner, tracking_url.
        On failure: found=False and an 'error' key.
    """
    try:
        client = get_client()
        result = (
            client.table("orders")
            .select("*")
            .eq("order_id", order_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            return {
                "found": False,
                "order_id": order_id,
                "error": f"No order found with id={order_id} for this user.",
            }
        row = result.data[0]
        return {
            "found": True,
            "order_id": row.get("order_id", order_id),
            "status": row.get("status", "unknown"),
            "items": row.get("items", []),
            "amount_inr": row.get("amount_inr", 0.0),
            "placed_at": str(row.get("placed_at", "")),
            "estimated_delivery": str(row.get("estimated_delivery", "")),
            "delivery_partner": row.get("delivery_partner"),
            "tracking_url": row.get("tracking_url"),
        }
    except Exception as exc:
        logger.error("order_lookup_tool failed for order=%s: %s", order_id, exc)
        return {"found": False, "order_id": order_id, "error": str(exc)}
