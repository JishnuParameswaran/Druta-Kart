"""
Druta Kart - Customer churn-risk scorer.

Produces a deterministic, rule-based ChurnRiskScore from a CustomerProfile.
No LLM call is made here; this is fast, auditable, and testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from db.models import CustomerProfile, CustomerSegment


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

@dataclass
class ChurnRiskScore:
    """Result returned by score_churn_risk().

    Attributes:
        score:              Float in [0.0, 1.0].  0 = no risk, 1 = certain churn.
        risk_level:         Human-readable bucket: "low" | "medium" | "high" | "critical".
        factors:            List of plain-English strings explaining each contribution.
        recommended_action: One-line suggestion for the retention agent.
    """
    score: float
    risk_level: str
    factors: List[str] = field(default_factory=list)
    recommended_action: str = ""


# ---------------------------------------------------------------------------
# Scoring weights (sum of max contributions = 1.0)
# ---------------------------------------------------------------------------

# Segment base scores
_SEGMENT_SCORES: dict[str, float] = {
    CustomerSegment.churning.value:             0.45,
    CustomerSegment.frequent_complainer.value:  0.35,
    CustomerSegment.bulk.value:                 0.15,  # high-value, worth monitoring
    CustomerSegment.regular.value:              0.10,
    CustomerSegment.new.value:                  0.10,
}

# Satisfaction score contribution (max 0.25 downward pressure)
_SAT_THRESHOLDS = [
    (2.0, 0.25, "very low satisfaction score"),
    (3.0, 0.15, "below-average satisfaction score"),
    (3.5, 0.05, "moderate satisfaction score"),
]

# Complaint count contribution (max 0.20)
_COMPLAINT_THRESHOLDS = [
    (6, 0.20, "6+ complaints on record"),
    (4, 0.14, "4–5 complaints on record"),
    (2, 0.07, "2–3 complaints on record"),
]

# Days-since-last-complaint contribution (max 0.10)
_RECENCY_THRESHOLDS = [
    (60, 0.10, "no activity in 60+ days"),
    (30, 0.06, "no activity in 30–59 days"),
    (14, 0.02, "no activity in 14–29 days"),
]

# Risk-level buckets
_RISK_LEVELS = [
    (0.70, "critical"),
    (0.50, "high"),
    (0.30, "medium"),
    (0.00, "low"),
]

# Recommended actions keyed by risk level
_ACTIONS: dict[str, str] = {
    "critical": "Immediately offer maximum retention incentive and escalate to human agent if unresolved.",
    "high":     "Proactively offer wallet credit or discount before the customer asks.",
    "medium":   "Acknowledge the inconvenience and offer a goodwill discount or priority support.",
    "low":      "Thank the customer and offer a small loyalty benefit to strengthen the relationship.",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_churn_risk(profile: CustomerProfile) -> ChurnRiskScore:
    """Compute a churn-risk score for *profile*.

    The score is the clipped sum of independent factor contributions.
    All arithmetic is deterministic and LLM-free.

    Args:
        profile: A fully-populated CustomerProfile instance.

    Returns:
        ChurnRiskScore with score, risk_level, factors, and recommended_action.
    """
    raw_score: float = 0.0
    factors: List[str] = []

    # 1. Segment base score
    segment = str(profile.customer_segment)
    seg_contribution = _SEGMENT_SCORES.get(segment, 0.10)
    raw_score += seg_contribution
    factors.append(f"segment={segment} (+{seg_contribution:.2f})")

    # 2. Satisfaction score
    if profile.satisfaction_score is not None:
        for threshold, contribution, label in _SAT_THRESHOLDS:
            if profile.satisfaction_score < threshold:
                raw_score += contribution
                factors.append(f"{label} (+{contribution:.2f})")
                break
        else:
            # satisfaction >= 3.5 — slightly reduce score
            raw_score = max(0.0, raw_score - 0.05)
            factors.append("good satisfaction score (-0.05)")

    # 3. Complaint count
    for threshold, contribution, label in _COMPLAINT_THRESHOLDS:
        if profile.complaint_count >= threshold:
            raw_score += contribution
            factors.append(f"{label} (+{contribution:.2f})")
            break

    # 4. Recency (days since last complaint / activity)
    if profile.last_complaint_date is not None:
        last = profile.last_complaint_date
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        days_inactive = (datetime.now(timezone.utc) - last).days
        for threshold, contribution, label in _RECENCY_THRESHOLDS:
            if days_inactive >= threshold:
                raw_score += contribution
                factors.append(f"{label} ({days_inactive}d, +{contribution:.2f})")
                break
    else:
        # No complaint date means either brand new or no history → mild risk
        raw_score += 0.03
        factors.append("no complaint history (+0.03)")

    # 5. Clip to [0, 1]
    score = round(min(1.0, max(0.0, raw_score)), 4)

    # 6. Bucket into risk level
    risk_level = "low"
    for threshold, level in _RISK_LEVELS:
        if score >= threshold:
            risk_level = level
            break

    return ChurnRiskScore(
        score=score,
        risk_level=risk_level,
        factors=factors,
        recommended_action=_ACTIONS[risk_level],
    )
