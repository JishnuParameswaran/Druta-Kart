"""
Druta Kart - Order Agent.

Handles: late delivery, order tracking, ETA queries, cancellation requests.
Calls order_lookup_tool for real-time status.
For orders delayed beyond SLA, auto-applies ₹50 wallet credit.
"""
from __future__ import annotations

import logging
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

_LATE_ORDER_CREDIT_INR = 50.0
_LATE_THRESHOLD_MINUTES = 30  # minutes beyond promised ETA


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=settings.groq_text_model,
        api_key=settings.groq_api_key,
        temperature=0.3,
    )


def _get_system_prompt() -> str:
    try:
        from prompts.prompt_registry import get_prompt
        return get_prompt("system_prompt")
    except Exception:
        return "You are a helpful customer support agent for Druta Kart."


def _is_late(order: dict) -> bool:
    """Heuristic: order is late if status contains 'delay' or estimated_delivery
    is in the past.  Real implementation would compare timestamps."""
    status = str(order.get("status", "")).lower()
    return "delay" in status or "late" in status or status == "delayed"


def _build_tracking_response(order: dict, user_message: str, language: str) -> str:
    """Ask the LLM to compose a tracking response from order data."""
    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage, SystemMessage

        order_info = (
            f"Order ID: {order.get('order_id', 'N/A')}\n"
            f"Status: {order.get('status', 'unknown')}\n"
            f"Items: {', '.join(order.get('items', [])) or 'N/A'}\n"
            f"Amount: ₹{order.get('amount_inr', 0)}\n"
            f"Estimated delivery: {order.get('estimated_delivery', 'N/A')}\n"
            f"Delivery partner: {order.get('delivery_partner', 'N/A')}\n"
            f"Tracking URL: {order.get('tracking_url', 'N/A')}"
        )

        prompt = (
            f"Customer asked: \"{user_message}\"\n\n"
            f"Order details:\n{order_info}\n\n"
            f"Respond in a warm, concise way in the customer's language ({language}). "
            f"Give the ETA and status clearly. If there is a tracking URL, mention it. "
            f"Do NOT invent any information not present in the order details above."
        )

        result = llm.invoke([
            SystemMessage(content=_get_system_prompt()),
            HumanMessage(content=prompt),
        ])
        return result.content.strip()

    except Exception as exc:
        logger.error("Order tracking response generation failed: %s", exc)
        status = order.get("status", "unknown")
        eta = order.get("estimated_delivery", "soon")
        return (
            f"Your order status is: {status}. Estimated delivery: {eta}. "
            f"We apologise for any inconvenience."
        )


def _build_not_found_response(order_id: str, user_message: str) -> str:
    return (
        f"I was unable to find order {order_id} linked to your account. "
        f"Please double-check the order ID and try again. If the issue persists, "
        f"our team will be happy to investigate further."
    )


def _build_general_order_response(user_message: str, language: str) -> str:
    """Fallback when no order_id is provided."""
    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage, SystemMessage
        result = llm.invoke([
            SystemMessage(content=_get_system_prompt()),
            HumanMessage(content=(
                f"Customer message: \"{user_message}\"\n"
                f"The customer seems to have an order-related query but did not provide "
                f"an order ID. Politely ask for the order ID to look up the status. "
                f"Respond in {language}."
            )),
        ])
        return result.content.strip()
    except Exception as exc:
        logger.error("General order response failed: %s", exc)
        return (
            "Could you please share your order ID so I can look up the details for you?"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(state: dict) -> dict:
    """Handle order tracking, late delivery, and cancellation queries.

    Args:
        state: AgentState dict.

    Returns:
        Partial AgentState update.
    """
    user_id = state.get("user_id", "")
    session_id = state.get("session_id", "")
    language = state.get("language", "en-IN")
    order_id: Optional[str] = state.get("order_id")
    tools_called = list(state.get("tools_called", []))

    # Extract last user message
    messages = state.get("messages", [])
    user_message = ""
    for msg in reversed(messages):
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        if role in ("human", "user"):
            user_message = getattr(msg, "content", str(msg))
            break

    # No order ID → ask for it
    if not order_id:
        response = _build_general_order_response(user_message, language)
        return {
            "response": response,
            "resolved": False,
            "tools_called": tools_called,
        }

    # Look up the order
    try:
        from tools.order_lookup_tool import order_lookup_tool
        order = order_lookup_tool.invoke({"order_id": order_id, "user_id": user_id})
        tools_called.append("order_lookup_tool")
    except Exception as exc:
        logger.error("order_lookup_tool failed: %s", exc)
        return {
            "response": (
                "I'm having trouble retrieving your order details right now. "
                "Please try again in a moment or contact support."
            ),
            "resolved": False,
            "tools_called": tools_called,
        }

    if not order.get("found"):
        return {
            "response": _build_not_found_response(order_id, user_message),
            "resolved": False,
            "tools_called": tools_called,
        }

    # Auto wallet credit for late orders
    offer_given = None
    if _is_late(order):
        try:
            from tools.wallet_credit_tool import wallet_credit_tool
            credit = wallet_credit_tool.invoke({
                "user_id": user_id,
                "amount_inr": _LATE_ORDER_CREDIT_INR,
                "reason": f"SLA breach compensation for late order {order_id}",
            })
            tools_called.append("wallet_credit_tool")
            offer_given = {
                "offer_type": "wallet_credit",
                "offer_value": credit.get("credited_amount", _LATE_ORDER_CREDIT_INR),
                "offer_description": credit.get("message", ""),
            }
            logger.info("Late order wallet credit applied: user=%s order=%s", user_id, order_id)
        except Exception as exc:
            logger.error("wallet_credit_tool failed for late order: %s", exc)

    # Build tracking response
    response = _build_tracking_response(order, user_message, language)

    # Append late credit note if applied
    if offer_given:
        response += (
            f"\n\nWe're sorry for the delay! As compensation, ₹{int(offer_given['offer_value'])} "
            f"has been added to your Druta Kart wallet."
        )

    return {
        "response": response,
        "resolved": True,
        "offer_given": offer_given,
        "tools_called": tools_called,
    }
