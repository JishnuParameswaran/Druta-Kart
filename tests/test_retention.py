"""
Tests for backend/retention/profile_scorer.py and offer_engine.py.

All logic is deterministic and LLM-free, so no mocking of external services
is needed.  Safety-cap tests patch settings attributes in-place.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

# conftest.py adds backend/ to sys.path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile(**kwargs):
    from db.models import CustomerProfile
    defaults = dict(user_id="u1", name="Test User")
    defaults.update(kwargs)
    return CustomerProfile(**defaults)


def _risk(score: float, risk_level: str):
    from retention.profile_scorer import ChurnRiskScore
    return ChurnRiskScore(score=score, risk_level=risk_level)


# ===========================================================================
# profile_scorer — ChurnRiskScore dataclass
# ===========================================================================

class TestChurnRiskScoreDataclass:

    def test_fields_accessible(self):
        from retention.profile_scorer import ChurnRiskScore
        r = ChurnRiskScore(score=0.5, risk_level="high")
        assert r.score == 0.5
        assert r.risk_level == "high"
        assert r.factors == []
        assert r.recommended_action == ""

    def test_factors_default_is_independent(self):
        from retention.profile_scorer import ChurnRiskScore
        r1 = ChurnRiskScore(score=0.1, risk_level="low")
        r2 = ChurnRiskScore(score=0.2, risk_level="low")
        r1.factors.append("x")
        assert r2.factors == []


# ===========================================================================
# profile_scorer — score_churn_risk
# ===========================================================================

class TestScoreChurnRisk:

    # --- return type & invariants ---

    def test_returns_churn_risk_score(self):
        from retention.profile_scorer import score_churn_risk, ChurnRiskScore
        result = score_churn_risk(_profile())
        assert isinstance(result, ChurnRiskScore)

    def test_score_always_in_range(self):
        from retention.profile_scorer import score_churn_risk
        profiles = [
            _profile(customer_segment="churning", complaint_count=10, satisfaction_score=1.0),
            _profile(customer_segment="new", complaint_count=0, satisfaction_score=5.0),
            _profile(customer_segment="regular"),
        ]
        for p in profiles:
            r = score_churn_risk(p)
            assert 0.0 <= r.score <= 1.0, f"score out of range: {r.score}"

    def test_factors_non_empty(self):
        from retention.profile_scorer import score_churn_risk
        r = score_churn_risk(_profile())
        assert len(r.factors) >= 1

    def test_recommended_action_populated(self):
        from retention.profile_scorer import score_churn_risk
        r = score_churn_risk(_profile())
        assert len(r.recommended_action) > 0

    # --- segment base scores ---

    def test_churning_higher_than_new(self):
        from retention.profile_scorer import score_churn_risk
        churning = score_churn_risk(_profile(customer_segment="churning"))
        new = score_churn_risk(_profile(customer_segment="new"))
        assert churning.score > new.score

    def test_frequent_complainer_higher_than_regular(self):
        from retention.profile_scorer import score_churn_risk
        freq = score_churn_risk(_profile(customer_segment="frequent_complainer"))
        reg  = score_churn_risk(_profile(customer_segment="regular"))
        assert freq.score > reg.score

    # --- satisfaction contributions ---

    def test_very_low_satisfaction_increases_score(self):
        from retention.profile_scorer import score_churn_risk
        low  = score_churn_risk(_profile(satisfaction_score=1.5))
        high = score_churn_risk(_profile(satisfaction_score=4.5))
        assert low.score > high.score

    def test_good_satisfaction_reduces_score(self):
        from retention.profile_scorer import score_churn_risk
        no_sat  = score_churn_risk(_profile(satisfaction_score=None))
        good    = score_churn_risk(_profile(satisfaction_score=4.5))
        # good satisfaction should reduce relative to no info (which adds 0.03 for no history)
        # The good path subtracts 0.05, no-sat path adds 0.03 for no history
        # Both start at the same segment base; good should be lower
        assert good.score < no_sat.score

    def test_sat_below_2_adds_max_contribution(self):
        from retention.profile_scorer import score_churn_risk
        r = score_churn_risk(_profile(satisfaction_score=1.0))
        assert any("very low satisfaction" in f for f in r.factors)

    def test_sat_between_2_and_3_adds_medium_contribution(self):
        from retention.profile_scorer import score_churn_risk
        r = score_churn_risk(_profile(satisfaction_score=2.5))
        assert any("below-average satisfaction" in f for f in r.factors)

    def test_sat_above_3_5_gives_good_label(self):
        from retention.profile_scorer import score_churn_risk
        r = score_churn_risk(_profile(satisfaction_score=4.0))
        assert any("good satisfaction" in f for f in r.factors)

    # --- complaint count contributions ---

    def test_high_complaint_count_increases_score(self):
        from retention.profile_scorer import score_churn_risk
        many = score_churn_risk(_profile(complaint_count=8))
        none = score_churn_risk(_profile(complaint_count=0))
        assert many.score > none.score

    def test_6_plus_complaints_label(self):
        from retention.profile_scorer import score_churn_risk
        r = score_churn_risk(_profile(complaint_count=6))
        assert any("6+" in f for f in r.factors)

    def test_4_to_5_complaints_label(self):
        from retention.profile_scorer import score_churn_risk
        r = score_churn_risk(_profile(complaint_count=4))
        assert any("4–5" in f for f in r.factors)

    def test_2_to_3_complaints_label(self):
        from retention.profile_scorer import score_churn_risk
        r = score_churn_risk(_profile(complaint_count=2))
        assert any("2–3" in f for f in r.factors)

    # --- recency contributions ---

    def test_old_last_complaint_increases_score(self):
        from retention.profile_scorer import score_churn_risk
        old    = _profile(last_complaint_date=datetime.now(timezone.utc) - timedelta(days=90))
        recent = _profile(last_complaint_date=datetime.now(timezone.utc) - timedelta(days=2))
        assert score_churn_risk(old).score > score_churn_risk(recent).score

    def test_60_plus_days_label(self):
        from retention.profile_scorer import score_churn_risk
        p = _profile(last_complaint_date=datetime.now(timezone.utc) - timedelta(days=65))
        r = score_churn_risk(p)
        assert any("60+" in f for f in r.factors)

    def test_30_to_59_days_label(self):
        from retention.profile_scorer import score_churn_risk
        p = _profile(last_complaint_date=datetime.now(timezone.utc) - timedelta(days=40))
        r = score_churn_risk(p)
        assert any("30–59" in f for f in r.factors)

    def test_14_to_29_days_label(self):
        from retention.profile_scorer import score_churn_risk
        p = _profile(last_complaint_date=datetime.now(timezone.utc) - timedelta(days=20))
        r = score_churn_risk(p)
        assert any("14–29" in f for f in r.factors)

    def test_no_complaint_date_adds_mild_score(self):
        from retention.profile_scorer import score_churn_risk
        r = score_churn_risk(_profile(last_complaint_date=None))
        assert any("no complaint history" in f for f in r.factors)

    def test_naive_datetime_handled(self):
        """Naive datetimes (no tzinfo) should not raise."""
        from retention.profile_scorer import score_churn_risk
        p = _profile(last_complaint_date=datetime.utcnow() - timedelta(days=45))
        r = score_churn_risk(p)
        assert 0.0 <= r.score <= 1.0

    # --- risk level bucketing ---

    def test_risk_level_critical_at_high_score(self):
        from retention.profile_scorer import score_churn_risk
        p = _profile(
            customer_segment="churning",
            complaint_count=8,
            satisfaction_score=1.0,
            last_complaint_date=datetime.now(timezone.utc) - timedelta(days=90),
        )
        r = score_churn_risk(p)
        assert r.risk_level == "critical"

    def test_risk_level_low_for_happy_new_customer(self):
        from retention.profile_scorer import score_churn_risk
        p = _profile(
            customer_segment="new",
            complaint_count=0,
            satisfaction_score=5.0,
            last_complaint_date=datetime.now(timezone.utc) - timedelta(days=1),
        )
        r = score_churn_risk(p)
        assert r.risk_level in ("low", "medium")

    def test_risk_levels_are_valid_strings(self):
        from retention.profile_scorer import score_churn_risk
        for seg in ("new", "regular", "bulk", "churning", "frequent_complainer"):
            r = score_churn_risk(_profile(customer_segment=seg))
            assert r.risk_level in ("low", "medium", "high", "critical")

    def test_score_does_not_exceed_1_under_extreme_input(self):
        from retention.profile_scorer import score_churn_risk
        p = _profile(
            customer_segment="churning",
            complaint_count=100,
            satisfaction_score=0.0,
            last_complaint_date=datetime.now(timezone.utc) - timedelta(days=365),
        )
        r = score_churn_risk(p)
        assert r.score == 1.0


# ===========================================================================
# offer_engine — safety clamp helpers
# ===========================================================================

class TestClampHelpers:

    def test_clamp_wallet_at_cap(self):
        from retention.offer_engine import _clamp_wallet
        from config import settings
        assert _clamp_wallet(settings.max_wallet_credit_inr + 500) == float(settings.max_wallet_credit_inr)

    def test_clamp_wallet_below_cap(self):
        from retention.offer_engine import _clamp_wallet
        assert _clamp_wallet(50.0) == 50.0

    def test_clamp_wallet_negative(self):
        from retention.offer_engine import _clamp_wallet
        assert _clamp_wallet(-10.0) == 0.0

    def test_clamp_discount_at_cap(self):
        from retention.offer_engine import _clamp_discount
        from config import settings
        assert _clamp_discount(settings.max_discount_percent + 20) == float(settings.max_discount_percent)

    def test_clamp_discount_below_cap(self):
        from retention.offer_engine import _clamp_discount
        assert _clamp_discount(10.0) == 10.0

    def test_clamp_free_items_at_cap(self):
        from retention.offer_engine import _clamp_free_items
        from config import settings
        assert _clamp_free_items(settings.max_free_items_per_complaint + 5) == settings.max_free_items_per_complaint

    def test_clamp_free_items_zero(self):
        from retention.offer_engine import _clamp_free_items
        assert _clamp_free_items(0) == 0


# ===========================================================================
# offer_engine — generate_offer matrix
# ===========================================================================

class TestGenerateOffer:

    # --- return type ---

    def test_returns_retention_offer(self):
        from retention.offer_engine import generate_offer, RetentionOffer
        offer = generate_offer(_profile(customer_segment="regular"), _risk(0.55, "high"))
        assert isinstance(offer, RetentionOffer)

    def test_offer_description_non_empty(self):
        from retention.offer_engine import generate_offer
        offer = generate_offer(_profile(customer_segment="new"), _risk(0.2, "low"))
        assert len(offer.offer_description) > 0

    def test_reasoning_non_empty(self):
        from retention.offer_engine import generate_offer
        offer = generate_offer(_profile(customer_segment="bulk"), _risk(0.6, "high"))
        assert len(offer.reasoning) > 0

    # --- segment/risk matrix spot-checks ---

    def test_churning_critical_gives_wallet_credit(self):
        from retention.offer_engine import generate_offer
        offer = generate_offer(_profile(customer_segment="churning"), _risk(0.8, "critical"))
        assert offer.offer_type == "wallet_credit"

    def test_churning_medium_gives_discount(self):
        from retention.offer_engine import generate_offer
        offer = generate_offer(_profile(customer_segment="churning"), _risk(0.35, "medium"))
        assert offer.offer_type == "discount"

    def test_bulk_always_gives_discount(self):
        from retention.offer_engine import generate_offer
        for risk_level in ("low", "medium", "high", "critical"):
            offer = generate_offer(_profile(customer_segment="bulk"), _risk(0.5, risk_level))
            assert offer.offer_type == "discount", f"expected discount for bulk/{risk_level}"

    def test_new_always_gives_wallet_credit(self):
        from retention.offer_engine import generate_offer
        for risk_level in ("low", "medium", "high", "critical"):
            offer = generate_offer(_profile(customer_segment="new"), _risk(0.3, risk_level))
            assert offer.offer_type == "wallet_credit", f"expected wallet_credit for new/{risk_level}"

    def test_frequent_complainer_critical_gives_priority_support(self):
        from retention.offer_engine import generate_offer
        offer = generate_offer(
            _profile(customer_segment="frequent_complainer"),
            _risk(0.9, "critical"),
        )
        assert offer.offer_type == "priority_support"
        assert offer.offer_value == 0.0

    def test_frequent_complainer_high_gives_wallet_not_free_item(self):
        """Frequent complainers must NOT get free items (moral hazard)."""
        from retention.offer_engine import generate_offer
        offer = generate_offer(
            _profile(customer_segment="frequent_complainer"),
            _risk(0.6, "high"),
            complaint_type="damaged",
        )
        assert offer.offer_type != "free_item"

    # --- free-item upgrade logic ---

    def test_damaged_complaint_high_risk_non_freq_gets_free_item(self):
        from retention.offer_engine import generate_offer
        offer = generate_offer(
            _profile(customer_segment="regular"),
            _risk(0.6, "high"),
            complaint_type="damaged",
        )
        assert offer.offer_type == "free_item"

    def test_missing_complaint_critical_risk_gets_free_item(self):
        from retention.offer_engine import generate_offer
        offer = generate_offer(
            _profile(customer_segment="churning"),
            _risk(0.8, "critical"),
            complaint_type="missing",
        )
        assert offer.offer_type == "free_item"

    def test_wrong_complaint_high_risk_gets_free_item(self):
        from retention.offer_engine import generate_offer
        offer = generate_offer(
            _profile(customer_segment="new"),
            _risk(0.55, "high"),
            complaint_type="wrong",
        )
        assert offer.offer_type == "free_item"

    def test_physical_complaint_low_risk_no_free_item(self):
        """Low risk → free item upgrade should NOT trigger."""
        from retention.offer_engine import generate_offer
        offer = generate_offer(
            _profile(customer_segment="regular"),
            _risk(0.1, "low"),
            complaint_type="damaged",
        )
        assert offer.offer_type != "free_item"

    def test_physical_complaint_medium_risk_no_free_item(self):
        """Medium risk → free item upgrade should NOT trigger."""
        from retention.offer_engine import generate_offer
        offer = generate_offer(
            _profile(customer_segment="regular"),
            _risk(0.35, "medium"),
            complaint_type="damaged",
        )
        assert offer.offer_type != "free_item"

    def test_non_physical_complaint_no_free_item(self):
        """'late' complaint should not trigger free-item upgrade."""
        from retention.offer_engine import generate_offer
        offer = generate_offer(
            _profile(customer_segment="regular"),
            _risk(0.7, "critical"),
            complaint_type="late",
        )
        assert offer.offer_type != "free_item"

    def test_no_complaint_type_uses_matrix(self):
        from retention.offer_engine import generate_offer
        offer = generate_offer(
            _profile(customer_segment="regular"),
            _risk(0.6, "high"),
            complaint_type=None,
        )
        # Should use matrix → regular/high → wallet_credit
        assert offer.offer_type == "wallet_credit"

    # --- safety caps enforced ---

    def test_wallet_value_never_exceeds_cap(self):
        from retention.offer_engine import generate_offer
        from config import settings
        for seg in ("churning", "regular", "new"):
            for rl in ("low", "medium", "high", "critical"):
                offer = generate_offer(_profile(customer_segment=seg), _risk(0.5, rl))
                if offer.offer_type == "wallet_credit":
                    assert offer.offer_value <= settings.max_wallet_credit_inr, (
                        f"{seg}/{rl}: wallet {offer.offer_value} > cap {settings.max_wallet_credit_inr}"
                    )

    def test_discount_value_never_exceeds_cap(self):
        from retention.offer_engine import generate_offer
        from config import settings
        for seg in ("churning", "bulk", "frequent_complainer"):
            for rl in ("low", "medium", "high", "critical"):
                offer = generate_offer(_profile(customer_segment=seg), _risk(0.5, rl))
                if offer.offer_type == "discount":
                    assert offer.offer_value <= settings.max_discount_percent, (
                        f"{seg}/{rl}: discount {offer.offer_value}% > cap {settings.max_discount_percent}%"
                    )

    def test_free_item_value_never_exceeds_cap(self):
        from retention.offer_engine import generate_offer
        from config import settings
        offer = generate_offer(
            _profile(customer_segment="regular"),
            _risk(0.8, "critical"),
            complaint_type="damaged",
        )
        assert offer.offer_value <= settings.max_free_items_per_complaint

    def test_wallet_cap_enforced_when_settings_lowered(self):
        """Patching settings.max_wallet_credit_inr to 50 should cap wallet at 50."""
        from retention.offer_engine import generate_offer
        import config
        with patch.object(config.settings, "max_wallet_credit_inr", 50):
            offer = generate_offer(
                _profile(customer_segment="churning"),
                _risk(0.8, "critical"),
            )
        if offer.offer_type == "wallet_credit":
            assert offer.offer_value <= 50

    def test_discount_cap_enforced_when_settings_lowered(self):
        from retention.offer_engine import generate_offer
        import config
        with patch.object(config.settings, "max_discount_percent", 5):
            offer = generate_offer(
                _profile(customer_segment="bulk"),
                _risk(0.8, "critical"),
            )
        if offer.offer_type == "discount":
            assert offer.offer_value <= 5

    # --- unknown segment falls back gracefully ---

    def test_unknown_segment_falls_back_to_default(self):
        from retention.offer_engine import generate_offer
        # Force an unrecognised segment string by creating profile and overriding in model_copy
        p = _profile(customer_segment="regular")
        # Manually set a bogus segment string via model_copy
        p2 = p.model_copy(update={"customer_segment": "vip_tier_unknown"})
        offer = generate_offer(p2, _risk(0.4, "medium"))
        # Should not raise; should return some offer
        assert offer.offer_type in ("wallet_credit", "discount", "free_item", "priority_support")


# ===========================================================================
# offer_engine — to_offer_given
# ===========================================================================

class TestToOfferGiven:

    def test_returns_offer_given_model(self):
        from retention.offer_engine import generate_offer, to_offer_given
        from db.models import OfferGiven
        offer = generate_offer(_profile(customer_segment="regular"), _risk(0.5, "high"))
        og = to_offer_given(offer, user_id="u1", session_id="s1")
        assert isinstance(og, OfferGiven)

    def test_fields_copied_correctly(self):
        from retention.offer_engine import generate_offer, to_offer_given
        offer = generate_offer(_profile(customer_segment="new"), _risk(0.2, "low"))
        og = to_offer_given(offer, user_id="u42", session_id="sess-99")
        assert og.user_id == "u42"
        assert og.session_id == "sess-99"
        assert og.offer_type == offer.offer_type
        assert og.offer_value == offer.offer_value
        assert og.offer_description == offer.offer_description

    def test_accepted_defaults_false(self):
        from retention.offer_engine import generate_offer, to_offer_given
        offer = generate_offer(_profile(customer_segment="bulk"), _risk(0.6, "high"))
        og = to_offer_given(offer, "u1", "s1")
        assert og.accepted is False

    def test_offer_id_autogenerated(self):
        from retention.offer_engine import generate_offer, to_offer_given
        offer = generate_offer(_profile(customer_segment="regular"), _risk(0.4, "medium"))
        og1 = to_offer_given(offer, "u1", "s1")
        og2 = to_offer_given(offer, "u1", "s1")
        assert og1.offer_id != og2.offer_id


# ===========================================================================
# Integration: scorer → engine pipeline
# ===========================================================================

class TestScorerToEnginePipeline:

    def test_full_pipeline_no_exception(self):
        from retention.profile_scorer import score_churn_risk
        from retention.offer_engine import generate_offer, to_offer_given
        p = _profile(
            customer_segment="churning",
            complaint_count=5,
            satisfaction_score=2.5,
            last_complaint_date=datetime.now(timezone.utc) - timedelta(days=35),
        )
        risk = score_churn_risk(p)
        offer = generate_offer(p, risk, complaint_type="damaged")
        og = to_offer_given(offer, user_id=p.user_id, session_id="s1")
        assert og.offer_type in ("wallet_credit", "discount", "free_item", "priority_support")

    def test_high_risk_customer_gets_meaningful_offer(self):
        """Score >= 0.5 should yield high/critical risk and a non-trivial offer."""
        from retention.profile_scorer import score_churn_risk
        from retention.offer_engine import generate_offer
        p = _profile(
            customer_segment="churning",
            complaint_count=7,
            satisfaction_score=1.5,
            last_complaint_date=datetime.now(timezone.utc) - timedelta(days=70),
        )
        risk = score_churn_risk(p)
        assert risk.risk_level in ("high", "critical")
        offer = generate_offer(p, risk)
        assert offer.offer_value > 0 or offer.offer_type == "priority_support"

    def test_low_risk_new_happy_customer_gets_small_offer(self):
        from retention.profile_scorer import score_churn_risk
        from retention.offer_engine import generate_offer
        p = _profile(
            customer_segment="new",
            complaint_count=0,
            satisfaction_score=5.0,
            last_complaint_date=datetime.now(timezone.utc) - timedelta(days=1),
        )
        risk = score_churn_risk(p)
        offer = generate_offer(p, risk)
        # New + low risk → small wallet credit (≤ 50)
        assert offer.offer_type == "wallet_credit"
        assert offer.offer_value <= 50
