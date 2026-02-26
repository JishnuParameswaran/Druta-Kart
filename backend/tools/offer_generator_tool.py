"""
Druta Kart - Offer generator tool.

LangChain tool that scores a customer's churn risk and returns the best
personalised retention offer.  Delegates to the retention layer so all
safety caps and segmentation logic remain in one place.
"""
import logging
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool

from db.customer_repo import get_customer_profile
from db.supabase_client import get_client
from retention.offer_engine import generate_offer, to_offer_given
from retention.profile_scorer import score_churn_risk

logger = logging.getLogger(__name__)

# Fallback offer when the customer profile cannot be found
_DEFAULT_OFFER = {
    "offer_type": "wallet_credit",
    "offer_value": 50.0,
    "offer_description": "₹50 added to your Druta Kart wallet as a goodwill gesture.",
    "risk_level": "unknown",
    "risk_score": None,
}


@tool
def offer_generator_tool(
    user_id: str,
    session_id: str,
    complaint_type: Optional[str] = None,
) -> dict:
    """Generate a personalised retention offer for an at-risk customer.

    Fetches the customer profile, scores churn risk, and selects the best
    offer from the retention matrix.  The offer is persisted to the database
    before being returned.  All values are safety-capped.

    Use this when the customer is upset, considering cancelling, or after a
    complaint has been resolved and goodwill compensation is appropriate.

    Args:
        user_id:        The customer's user ID.
        session_id:     Current chat session ID (for DB logging).
        complaint_type: Optional complaint type ("damaged", "missing", "wrong",
                        "late", "payment") to allow physical-goods upgrade logic.

    Returns:
        Dict with keys: user_id, offer_type, offer_value, offer_description,
        risk_level, risk_score.  On failure: status="failed" and 'error'.
    """
    try:
        profile = get_customer_profile(user_id)
        if profile is None:
            logger.warning("offer_generator_tool: no profile for user=%s; using default offer", user_id)
            return {**_DEFAULT_OFFER, "user_id": user_id}

        risk = score_churn_risk(profile)
        offer = generate_offer(profile, risk, complaint_type=complaint_type)

        # Persist offer for analytics
        try:
            og = to_offer_given(offer, user_id=user_id, session_id=session_id)
            payload = og.model_dump()
            # Serialise datetime fields for Supabase
            for k, v in payload.items():
                if isinstance(v, datetime):
                    payload[k] = v.isoformat()
            get_client().table("offers_given").insert(payload).execute()
        except Exception as db_exc:
            logger.warning("offer_generator_tool: failed to persist offer: %s", db_exc)

        return {
            "user_id": user_id,
            "offer_type": offer.offer_type,
            "offer_value": offer.offer_value,
            "offer_description": offer.offer_description,
            "risk_level": risk.risk_level,
            "risk_score": risk.score,
        }
    except Exception as exc:
        logger.error("offer_generator_tool failed for user=%s: %s", user_id, exc)
        return {"user_id": user_id, "status": "failed", "error": str(exc)}
