"""
Druta Kart - Retention Agent.

Triggered after a complaint is resolved.  Scores churn risk, selects a
personalised retention offer, validates it against safety caps, and weaves
the offer naturally into the conversation response.
"""
from __future__ import annotations

import logging
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=settings.groq_text_model,
        api_key=settings.groq_api_key,
        temperature=0.4,
    )


def _validate_offer(offer: dict) -> dict:
    """Hard-validate offer values against safety caps.  Returns corrected dict."""
    offer_type = offer.get("offer_type", "")
    offer_value = float(offer.get("offer_value", 0))

    if offer_type == "wallet_credit":
        capped = min(offer_value, float(settings.max_wallet_credit_inr))
        if capped != offer_value:
            logger.warning(
                "Retention offer wallet_credit %.0f capped to %.0f",
                offer_value, capped,
            )
        offer["offer_value"] = capped
    elif offer_type == "discount":
        capped = min(offer_value, float(settings.max_discount_percent))
        if capped != offer_value:
            logger.warning(
                "Retention offer discount %.0f%% capped to %.0f%%",
                offer_value, capped,
            )
        offer["offer_value"] = capped
    elif offer_type == "free_item":
        capped = min(int(offer_value), settings.max_free_items_per_complaint)
        offer["offer_value"] = float(capped)

    return offer


def _build_offer_message(
    offer: dict,
    resolution_response: str,
    customer_name: str,
    emotion: str,
) -> str:
    """Ask the LLM to weave the offer naturally into the existing response."""
    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage

        offer_type = offer.get("offer_type", "")
        offer_value = offer.get("offer_value", 0)
        offer_desc = offer.get("offer_description", "")

        if offer_type == "wallet_credit":
            offer_summary = f"₹{int(offer_value)} wallet credit (instant)"
        elif offer_type == "discount":
            offer_summary = f"{int(offer_value)}% discount on next order (valid 7 days)"
        elif offer_type == "free_item":
            offer_summary = f"{int(offer_value)} complimentary item(s) on next order"
        elif offer_type == "priority_support":
            offer_summary = "priority support for your next interaction"
        else:
            offer_summary = offer_desc

        prompt = (
            f"You are a customer support agent for Druta Kart.\n\n"
            f"You have just resolved a complaint for {customer_name or 'the customer'} "
            f"who is feeling {emotion}.\n\n"
            f"Resolution message already sent:\n\"{resolution_response}\"\n\n"
            f"Now naturally add this goodwill offer at the end:\n"
            f"Offer: {offer_summary}\n\n"
            f"Write ONLY the complete updated message (resolution + offer naturally "
            f"integrated). Keep it warm, brief, and end with a positive note. "
            f"Do NOT add any offer ID, tracking number, or date that you are not sure of."
        )

        result = llm.invoke([HumanMessage(content=prompt)])
        return result.content.strip()

    except Exception as exc:
        logger.warning("Retention offer message generation failed: %s", exc)
        desc = offer.get("offer_description", "")
        return f"{resolution_response}\n\nAs a goodwill gesture, {desc}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(state: dict) -> dict:
    """Score churn risk and present a retention offer after complaint resolution.

    Args:
        state: AgentState dict with user_id, session_id, emotion, complaint_type,
               response (the resolution text from complaint_agent).

    Returns:
        Partial AgentState update with offer_given and updated response.
    """
    user_id = state.get("user_id", "")
    session_id = state.get("session_id", "")
    emotion = state.get("emotion", "neutral")
    complaint_type: Optional[str] = state.get("complaint_type")
    resolution_response = state.get("response", "")
    tools_called = list(state.get("tools_called", []))

    try:
        # Call offer_generator_tool (handles profile fetch + scorer internally)
        from tools.offer_generator_tool import offer_generator_tool
        offer_result = offer_generator_tool.invoke({
            "user_id": user_id,
            "session_id": session_id,
            "complaint_type": complaint_type,
        })
        tools_called.append("offer_generator_tool")

        if offer_result.get("status") == "failed":
            logger.warning("offer_generator_tool failed for user=%s", user_id)
            return {"tools_called": tools_called}

        # Validate caps — belt-and-suspenders on top of offer_engine
        offer = _validate_offer(offer_result)

        # Fetch customer name for personalisation
        customer_name = ""
        try:
            from db.customer_repo import get_customer_profile
            profile = get_customer_profile(user_id)
            if profile:
                customer_name = profile.name
        except Exception:
            pass

        # Weave offer into the resolution message
        updated_response = _build_offer_message(
            offer, resolution_response, customer_name, emotion
        )

        logger.info(
            "Retention offer presented: user=%s offer_type=%s value=%s risk=%s",
            user_id,
            offer.get("offer_type"),
            offer.get("offer_value"),
            offer.get("risk_level"),
        )

        return {
            "response": updated_response,
            "offer_given": offer,
            "tools_called": tools_called,
        }

    except Exception as exc:
        logger.error("Retention agent failed for user=%s: %s", user_id, exc)
        # Don't fail the whole conversation — return state unchanged
        return {"tools_called": tools_called}
