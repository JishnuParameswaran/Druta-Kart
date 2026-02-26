"""
Druta Kart - Retention offer engine.

Generates a RetentionOffer for a customer based on their churn-risk score,
segment, and (optionally) the complaint type that triggered the interaction.

SAFETY CAPS — all offer values are hard-clamped against config.py before
being returned.  The LLM never touches these values directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import settings
from db.models import CustomerProfile, CustomerSegment, OfferGiven
from retention.profile_scorer import ChurnRiskScore


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

@dataclass
class RetentionOffer:
    """An offer generated for a specific customer interaction.

    Attributes:
        offer_type:         "wallet_credit" | "discount" | "free_item" | "priority_support"
        offer_value:        Numeric value (₹ for wallet_credit, % for discount, count for free_item, 0 for support).
        offer_description:  Human-readable string ready to show to the customer.
        reasoning:          Internal note explaining why this offer was chosen (for logging/audit).
    """
    offer_type: str
    offer_value: float
    offer_description: str
    reasoning: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp_wallet(value: float) -> float:
    return min(float(settings.max_wallet_credit_inr), max(0.0, value))


def _clamp_discount(value: float) -> float:
    return min(float(settings.max_discount_percent), max(0.0, value))


def _clamp_free_items(value: int) -> int:
    return min(settings.max_free_items_per_complaint, max(0, value))


def _wallet_offer(amount: float, reasoning: str) -> RetentionOffer:
    capped = _clamp_wallet(amount)
    return RetentionOffer(
        offer_type="wallet_credit",
        offer_value=capped,
        offer_description=f"₹{int(capped)} added to your Druta Kart wallet as a goodwill gesture.",
        reasoning=reasoning,
    )


def _discount_offer(percent: float, reasoning: str) -> RetentionOffer:
    capped = _clamp_discount(percent)
    return RetentionOffer(
        offer_type="discount",
        offer_value=capped,
        offer_description=f"{int(capped)}% off on your next order — valid for 7 days.",
        reasoning=reasoning,
    )


def _free_item_offer(count: int, reasoning: str) -> RetentionOffer:
    capped = _clamp_free_items(count)
    return RetentionOffer(
        offer_type="free_item",
        offer_value=float(capped),
        offer_description=(
            f"{capped} complimentary item(s) added to your next order."
            if capped > 0 else "Complimentary item added to your next order."
        ),
        reasoning=reasoning,
    )


def _support_offer(reasoning: str) -> RetentionOffer:
    return RetentionOffer(
        offer_type="priority_support",
        offer_value=0.0,
        offer_description="You've been upgraded to priority support — a senior agent will assist you.",
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Offer selection matrix
# ---------------------------------------------------------------------------

# (segment, risk_level) → (offer_type, base_value)
# Values are pre-cap; _clamp_* is applied inside each helper.
_OFFER_MATRIX: dict[tuple[str, str], tuple[str, float]] = {
    # Churning customers — be aggressive
    ("churning",            "critical"): ("wallet_credit", 200),
    ("churning",            "high"):     ("wallet_credit", 150),
    ("churning",            "medium"):   ("discount",       25),
    ("churning",            "low"):      ("discount",       10),

    # Frequent complainers — avoid moral hazard; lean on support
    ("frequent_complainer", "critical"): ("priority_support", 0),
    ("frequent_complainer", "high"):     ("wallet_credit",   100),
    ("frequent_complainer", "medium"):   ("discount",         15),
    ("frequent_complainer", "low"):      ("discount",         10),

    # Bulk / high-value — discounts are more meaningful than fixed credits
    ("bulk",                "critical"): ("discount",  35),
    ("bulk",                "high"):     ("discount",  25),
    ("bulk",                "medium"):   ("discount",  15),
    ("bulk",                "low"):      ("discount",   5),

    # Regular
    ("regular",             "critical"): ("wallet_credit", 150),
    ("regular",             "high"):     ("wallet_credit", 100),
    ("regular",             "medium"):   ("wallet_credit",  75),
    ("regular",             "low"):      ("discount",       10),

    # New customers — small goodwill to build loyalty
    ("new",                 "critical"): ("wallet_credit", 100),
    ("new",                 "high"):     ("wallet_credit",  75),
    ("new",                 "medium"):   ("wallet_credit",  50),
    ("new",                 "low"):      ("wallet_credit",  25),
}

# Complaint-type overrides: certain complaint types warrant a free item
_COMPLAINT_FREE_ITEM_TYPES = {"damaged", "wrong", "missing"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_offer(
    profile: CustomerProfile,
    risk: ChurnRiskScore,
    complaint_type: Optional[str] = None,
) -> RetentionOffer:
    """Select and return the best RetentionOffer for this customer.

    The offer type and value are chosen from a rule matrix keyed by
    (segment, risk_level).  For physical-goods complaints (damaged / wrong /
    missing) the engine upgrades to a free-item offer when the risk is high
    enough.  All values are clamped by the safety caps in config.py.

    Args:
        profile:        CustomerProfile of the at-risk customer.
        risk:           ChurnRiskScore produced by profile_scorer.score_churn_risk().
        complaint_type: Optional complaint type string from ComplaintType enum
                        (e.g. "damaged", "missing").

    Returns:
        A RetentionOffer ready to surface to the customer.
    """
    segment = str(profile.customer_segment)
    risk_level = risk.risk_level

    # Physical-goods complaints at medium+ risk → free item (unless frequent complainer)
    if (
        complaint_type in _COMPLAINT_FREE_ITEM_TYPES
        and risk_level in ("high", "critical")
        and segment != CustomerSegment.frequent_complainer.value
    ):
        count = _clamp_free_items(1)
        return _free_item_offer(
            count,
            reasoning=(
                f"Free item offered for {complaint_type} complaint "
                f"(segment={segment}, risk={risk_level}, score={risk.score})"
            ),
        )

    # Matrix lookup with fallback to (regular, medium)
    offer_type, base_value = _OFFER_MATRIX.get(
        (segment, risk_level),
        _OFFER_MATRIX[("regular", "medium")],
    )

    reasoning = (
        f"Matrix lookup: segment={segment}, risk={risk_level}, "
        f"score={risk.score:.3f}, complaint_type={complaint_type}"
    )

    if offer_type == "wallet_credit":
        return _wallet_offer(base_value, reasoning)
    elif offer_type == "discount":
        return _discount_offer(base_value, reasoning)
    elif offer_type == "free_item":
        return _free_item_offer(int(base_value), reasoning)
    else:
        return _support_offer(reasoning)


def to_offer_given(
    offer: RetentionOffer,
    user_id: str,
    session_id: str,
) -> OfferGiven:
    """Convert a RetentionOffer into an OfferGiven db model for persistence.

    Args:
        offer:      The RetentionOffer returned by generate_offer().
        user_id:    Customer user_id.
        session_id: Current chat session id.

    Returns:
        An OfferGiven instance (not yet persisted — caller saves to DB).
    """
    return OfferGiven(
        user_id=user_id,
        session_id=session_id,
        offer_type=offer.offer_type,
        offer_value=offer.offer_value,
        offer_description=offer.offer_description,
        accepted=False,
    )
