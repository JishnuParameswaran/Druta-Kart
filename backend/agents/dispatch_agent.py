"""
Druta Kart - Dispatch Agent.

Handles delivery issues: not delivered, wrong address, late delivery,
damaged package, missing items, delivery partner unreachable, refused delivery.

Detects the issue type(s) from the customer message via keyword matching,
optionally calls order_lookup_tool when an order_id is available,
calls dispatch_checklist_tool to generate and persist an action checklist,
then uses the Groq LLM to compose an empathetic response.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Issue detection — keyword → issue_type mapping
# Keys must match the issue slugs accepted by dispatch_checklist_tool.
# ---------------------------------------------------------------------------

_ISSUE_KEYWORDS: dict[str, list[str]] = {
    "not_delivered": [
        "not delivered", "not received", "hasn't arrived", "havent received",
        "never came", "didn't deliver", "didnt deliver", "not yet delivered",
        "no delivery", "order not arrived", "not arrived", "still not here",
        "never arrived", "never delivered",
    ],
    "wrong_address": [
        "wrong address", "wrong location", "wrong house", "different address",
        "wrong flat", "wrong building", "delivered elsewhere", "wrong place",
        "neighbour got", "neighbor got", "delivered to wrong",
    ],
    "late_delivery": [
        "late", "delayed", "taking too long", "overdue", "slow delivery",
        "still waiting", "not on time", "past eta", "beyond eta", "hours late",
        "should have arrived", "expected by", "where is my order",
        "where's my order", "where is my delivery",
    ],
    "damaged_package": [
        "damaged", "broken", "crushed", "torn", "leaking", "open package",
        "tampered", "dented", "smashed", "spoiled", "bad condition",
        "damaged box", "damaged package",
    ],
    "missing_items": [
        "missing item", "item missing", "incomplete order", "partial order",
        "only received", "didn't get all", "didnt get all", "not all items",
        "left out", "forgot to include", "short order", "items not there",
    ],
    "delivery_partner_unreachable": [
        "delivery partner unreachable", "delivery guy unreachable",
        "delivery boy unreachable", "can't reach delivery", "cant reach delivery",
        "not picking up", "not answering", "delivery agent not responding",
        "driver unreachable", "no response from delivery", "delivery person not reachable",
    ],
    "refused_delivery": [
        "refused delivery", "reject delivery", "turned away", "didn't accept",
        "didnt accept", "returned without delivering", "refused the order",
        "delivery refused", "sent back", "return to sender",
    ],
}

_FALLBACK_ISSUE = "late_delivery"


def _detect_issues(text: str) -> List[str]:
    """
    Scan lowercased text for delivery issue keywords.

    Returns a deduplicated list of matched issue-type strings preserving
    the order they appear in _ISSUE_KEYWORDS.  Falls back to
    [_FALLBACK_ISSUE] when nothing matches.
    """
    lower = text.lower()
    found: list[str] = [
        issue_type
        for issue_type, keywords in _ISSUE_KEYWORDS.items()
        if any(kw in lower for kw in keywords)
    ]
    return found if found else [_FALLBACK_ISSUE]


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


def _build_dispatch_response(
    user_message: str,
    language: str,
    issues: List[str],
    checklist: dict,
    order: Optional[dict],
) -> str:
    """Ask the LLM to compose an empathetic dispatch escalation response."""
    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage, SystemMessage

        order_section = ""
        if order and order.get("found"):
            order_section = (
                f"Order details:\n"
                f"  Order ID: {order.get('order_id', 'N/A')}\n"
                f"  Status: {order.get('status', 'unknown')}\n"
                f"  Items: {', '.join(order.get('items', [])) or 'N/A'}\n"
                f"  Estimated delivery: {order.get('estimated_delivery', 'N/A')}\n"
                f"  Delivery partner: {order.get('delivery_partner', 'N/A')}\n\n"
            )

        issues_str = ", ".join(i.replace("_", " ") for i in issues)
        checklist_items = checklist.get("checklist_items", [])
        checklist_str = (
            "\n".join(f"  - {item}" for item in checklist_items)
            if checklist_items
            else "  - Investigating the issue with the dispatch team"
        )

        prompt = (
            f"Customer message: \"{user_message}\"\n"
            f"Reported delivery issue(s): {issues_str}\n\n"
            f"{order_section}"
            f"Actions our dispatch team is taking:\n{checklist_str}\n\n"
            f"Write a warm, concise customer support reply in {language}. "
            f"Acknowledge the specific issue with genuine empathy, "
            f"assure the customer the dispatch team has been notified and is acting now, "
            f"and give a realistic next step or expectation. "
            f"Do NOT promise a specific time unless order details show an ETA. "
            f"Do NOT invent any information not present above."
        )

        result = llm.invoke([
            SystemMessage(content=_get_system_prompt()),
            HumanMessage(content=prompt),
        ])
        return result.content.strip()

    except Exception as exc:
        logger.error("Dispatch response generation failed: %s", exc)
        return (
            "We're very sorry for the inconvenience with your delivery. "
            "Our dispatch team has been alerted and is actively looking into this. "
            "We'll keep you updated as soon as we have more information."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(state: dict) -> dict:
    """Handle delivery issues: detect issue type, trigger dispatch checklist,
    compose an empathetic LLM response.

    Args:
        state: AgentState dict.

    Returns:
        Partial AgentState update with response, resolved, resolution_type,
        and tools_called.
    """
    user_id      = state.get("user_id", "")
    session_id   = state.get("session_id", "")
    language     = state.get("language", "en-IN")
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

    # Detect delivery issue type(s) from message text
    issues = _detect_issues(user_message)
    logger.info(
        "Dispatch agent: user=%s issues=%s order_id=%s",
        user_id, issues, order_id,
    )

    # Optional: look up order details when order_id is present
    order: Optional[dict] = None
    if order_id:
        try:
            from tools.order_lookup_tool import order_lookup_tool
            order = order_lookup_tool.invoke({"order_id": order_id, "user_id": user_id})
            tools_called.append("order_lookup_tool")
        except Exception as exc:
            logger.error("order_lookup_tool failed in dispatch agent: %s", exc)

    # Generate and persist the dispatch action checklist
    checklist: dict = {}
    try:
        from tools.dispatch_checklist_tool import dispatch_checklist_tool
        checklist = dispatch_checklist_tool.invoke({
            "user_id": user_id,
            "session_id": session_id,
            "issues": issues,
        })
        tools_called.append("dispatch_checklist_tool")
        logger.info(
            "Dispatch checklist created: id=%s items=%d",
            checklist.get("checklist_id"), checklist.get("item_count", 0),
        )
    except Exception as exc:
        logger.error("dispatch_checklist_tool failed: %s", exc)

    # Build empathetic LLM response
    response = _build_dispatch_response(
        user_message=user_message,
        language=language,
        issues=issues,
        checklist=checklist,
        order=order,
    )

    return {
        "response": response,
        "resolved": True,
        "resolution_type": "dispatch_escalation",
        "tools_called": tools_called,
    }
