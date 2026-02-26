"""
Tests for backend/tools/.

All Supabase writes and customer-repo reads are mocked.
Each tool is invoked directly (not via LangChain chain) to test the
underlying Python function logic.

langchain_core is NOT installed in the bare test environment (it lives in
the Docker image), so we inject a minimal stub via sys.modules before any
tool module is imported.  The stub's @tool decorator adds .name and .invoke()
to the decorated function, which is all the tests need.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal langchain_core.tools stub — must happen before any tools import
# ---------------------------------------------------------------------------

def _make_tool_decorator():
    """Return a @tool decorator that adds .name and .invoke() to the function."""
    def tool(fn):
        fn.name = fn.__name__
        _orig = fn
        def invoke(input_dict: dict):
            return _orig(**input_dict)
        fn.invoke = invoke
        return fn
    return tool


def _inject_langchain_stub():
    lc_tools = ModuleType("langchain_core.tools")
    lc_tools.tool = _make_tool_decorator()      # type: ignore[attr-defined]

    lc_core = ModuleType("langchain_core")
    lc_core.tools = lc_tools                    # type: ignore[attr-defined]

    sys.modules.setdefault("langchain_core", lc_core)
    sys.modules.setdefault("langchain_core.tools", lc_tools)


_inject_langchain_stub()

# conftest.py adds backend/ to sys.path


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

def _supabase_ok():
    """Supabase client mock that accepts any insert/update and returns empty data."""
    mock_result = MagicMock()
    mock_result.data = []
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.limit.return_value = q
    q.insert.return_value = q
    q.update.return_value = q
    q.execute.return_value = mock_result
    client = MagicMock()
    client.table.return_value = q
    return client, q, mock_result


def _supabase_with_rows(rows):
    client, q, result = _supabase_ok()
    result.data = rows
    return client, q, result


# ===========================================================================
# order_lookup_tool
# ===========================================================================

class TestOrderLookupTool:

    def test_returns_dict(self):
        client, q, result = _supabase_ok()
        result.data = []
        with patch("tools.order_lookup_tool.get_client", return_value=client):
            from tools.order_lookup_tool import order_lookup_tool
            out = order_lookup_tool.invoke({"order_id": "ORD-1", "user_id": "u1"})
        assert isinstance(out, dict)

    def test_not_found_returns_found_false(self):
        client, q, result = _supabase_ok()
        result.data = []
        with patch("tools.order_lookup_tool.get_client", return_value=client):
            from tools.order_lookup_tool import order_lookup_tool
            out = order_lookup_tool.invoke({"order_id": "X", "user_id": "u1"})
        assert out["found"] is False
        assert "error" in out

    def test_found_order_returned(self):
        row = {
            "order_id": "ORD-42",
            "user_id": "u1",
            "status": "out_for_delivery",
            "items": ["Tomatoes 500g"],
            "amount_inr": 149.0,
            "placed_at": "2026-02-25T10:00:00",
            "estimated_delivery": "2026-02-25T10:30:00",
            "delivery_partner": "Swiggy Genie",
            "tracking_url": "https://track.example.com/ORD-42",
        }
        client, q, result = _supabase_with_rows([row])
        with patch("tools.order_lookup_tool.get_client", return_value=client):
            from tools.order_lookup_tool import order_lookup_tool
            out = order_lookup_tool.invoke({"order_id": "ORD-42", "user_id": "u1"})
        assert out["found"] is True
        assert out["status"] == "out_for_delivery"
        assert out["delivery_partner"] == "Swiggy Genie"

    def test_db_error_returns_found_false(self):
        with patch("tools.order_lookup_tool.get_client", side_effect=RuntimeError("DB down")):
            from tools.order_lookup_tool import order_lookup_tool
            out = order_lookup_tool.invoke({"order_id": "X", "user_id": "u1"})
        assert out["found"] is False
        assert "error" in out

    def test_tool_has_name(self):
        from tools.order_lookup_tool import order_lookup_tool
        assert order_lookup_tool.name == "order_lookup_tool"


# ===========================================================================
# refund_tool
# ===========================================================================

class TestRefundTool:

    def test_successful_refund(self):
        client, q, _ = _supabase_ok()
        with patch("tools.refund_tool.get_client", return_value=client):
            from tools.refund_tool import refund_tool
            out = refund_tool.invoke({
                "order_id": "ORD-1",
                "user_id": "u1",
                "amount_inr": 99.0,
                "reason": "item damaged",
            })
        assert out["status"] == "initiated"
        assert out["amount_inr"] == 99.0
        assert "refund_id" in out
        assert out["expected_days"] == 5

    def test_refund_id_is_unique(self):
        client, q, _ = _supabase_ok()
        with patch("tools.refund_tool.get_client", return_value=client):
            from tools.refund_tool import refund_tool
            r1 = refund_tool.invoke({"order_id": "O1", "user_id": "u1", "amount_inr": 50, "reason": "x"})
            r2 = refund_tool.invoke({"order_id": "O1", "user_id": "u1", "amount_inr": 50, "reason": "x"})
        assert r1["refund_id"] != r2["refund_id"]

    def test_message_contains_amount(self):
        client, q, _ = _supabase_ok()
        with patch("tools.refund_tool.get_client", return_value=client):
            from tools.refund_tool import refund_tool
            out = refund_tool.invoke({"order_id": "O1", "user_id": "u1", "amount_inr": 150, "reason": "x"})
        assert "150" in out["message"]

    def test_db_error_returns_failed(self):
        with patch("tools.refund_tool.get_client", side_effect=RuntimeError("DB down")):
            from tools.refund_tool import refund_tool
            out = refund_tool.invoke({"order_id": "O1", "user_id": "u1", "amount_inr": 50, "reason": "x"})
        assert out["status"] == "failed"
        assert "error" in out

    def test_amount_rounded_to_2dp(self):
        client, q, _ = _supabase_ok()
        with patch("tools.refund_tool.get_client", return_value=client):
            from tools.refund_tool import refund_tool
            out = refund_tool.invoke({"order_id": "O1", "user_id": "u1", "amount_inr": 99.999, "reason": "x"})
        assert out["amount_inr"] == round(99.999, 2)


# ===========================================================================
# replacement_tool
# ===========================================================================

class TestReplacementTool:

    def test_successful_replacement(self):
        client, q, _ = _supabase_ok()
        with patch("tools.replacement_tool.get_client", return_value=client):
            from tools.replacement_tool import replacement_tool
            out = replacement_tool.invoke({
                "order_id": "ORD-1",
                "user_id": "u1",
                "product_name": "Onions 1kg",
                "reason": "delivered wrong item",
            })
        assert out["status"] == "approved"
        assert out["product_name"] == "Onions 1kg"
        assert "replacement_id" in out

    def test_expected_delivery_is_tomorrow_or_later(self):
        client, q, _ = _supabase_ok()
        with patch("tools.replacement_tool.get_client", return_value=client):
            from tools.replacement_tool import replacement_tool
            out = replacement_tool.invoke({
                "order_id": "O1", "user_id": "u1",
                "product_name": "Milk 500ml", "reason": "missing",
            })
        expected = datetime.strptime(out["expected_delivery"], "%Y-%m-%d").date()
        today = datetime.utcnow().date()
        assert expected >= today

    def test_replacement_id_unique(self):
        client, q, _ = _supabase_ok()
        with patch("tools.replacement_tool.get_client", return_value=client):
            from tools.replacement_tool import replacement_tool
            r1 = replacement_tool.invoke({"order_id": "O1", "user_id": "u1", "product_name": "X", "reason": "y"})
            r2 = replacement_tool.invoke({"order_id": "O1", "user_id": "u1", "product_name": "X", "reason": "y"})
        assert r1["replacement_id"] != r2["replacement_id"]

    def test_message_contains_product_name(self):
        client, q, _ = _supabase_ok()
        with patch("tools.replacement_tool.get_client", return_value=client):
            from tools.replacement_tool import replacement_tool
            out = replacement_tool.invoke({
                "order_id": "O1", "user_id": "u1",
                "product_name": "Bread Loaf", "reason": "damaged",
            })
        assert "Bread Loaf" in out["message"]

    def test_db_error_returns_failed(self):
        with patch("tools.replacement_tool.get_client", side_effect=RuntimeError("DB down")):
            from tools.replacement_tool import replacement_tool
            out = replacement_tool.invoke({"order_id": "O1", "user_id": "u1", "product_name": "X", "reason": "y"})
        assert out["status"] == "failed"


# ===========================================================================
# wallet_credit_tool
# ===========================================================================

class TestWalletCreditTool:

    def test_successful_credit(self):
        client, q, _ = _supabase_ok()
        with patch("tools.wallet_credit_tool.get_client", return_value=client):
            from tools.wallet_credit_tool import wallet_credit_tool
            out = wallet_credit_tool.invoke({"user_id": "u1", "amount_inr": 100, "reason": "goodwill"})
        assert out["credited_amount"] == 100.0
        assert out["cap_applied"] is False

    def test_cap_enforced_above_max(self):
        client, q, _ = _supabase_ok()
        with patch("tools.wallet_credit_tool.get_client", return_value=client):
            from tools.wallet_credit_tool import wallet_credit_tool
            from config import settings
            out = wallet_credit_tool.invoke({
                "user_id": "u1",
                "amount_inr": settings.max_wallet_credit_inr + 500,
                "reason": "test",
            })
        assert out["credited_amount"] == float(settings.max_wallet_credit_inr)
        assert out["cap_applied"] is True

    def test_cap_enforced_when_settings_patched(self):
        client, q, _ = _supabase_ok()
        import config
        with patch("tools.wallet_credit_tool.get_client", return_value=client):
            with patch.object(config.settings, "max_wallet_credit_inr", 50):
                from tools.wallet_credit_tool import wallet_credit_tool
                out = wallet_credit_tool.invoke({"user_id": "u1", "amount_inr": 200, "reason": "test"})
        assert out["credited_amount"] <= 50

    def test_negative_amount_clamped_to_zero(self):
        client, q, _ = _supabase_ok()
        with patch("tools.wallet_credit_tool.get_client", return_value=client):
            from tools.wallet_credit_tool import wallet_credit_tool
            out = wallet_credit_tool.invoke({"user_id": "u1", "amount_inr": -50, "reason": "test"})
        assert out["credited_amount"] == 0.0

    def test_message_contains_amount(self):
        client, q, _ = _supabase_ok()
        with patch("tools.wallet_credit_tool.get_client", return_value=client):
            from tools.wallet_credit_tool import wallet_credit_tool
            out = wallet_credit_tool.invoke({"user_id": "u1", "amount_inr": 75, "reason": "test"})
        assert "75" in out["message"]

    def test_db_error_returns_failed(self):
        with patch("tools.wallet_credit_tool.get_client", side_effect=RuntimeError("down")):
            from tools.wallet_credit_tool import wallet_credit_tool
            out = wallet_credit_tool.invoke({"user_id": "u1", "amount_inr": 50, "reason": "x"})
        assert out["status"] == "failed"


# ===========================================================================
# offer_generator_tool
# ===========================================================================

class TestOfferGeneratorTool:

    def _make_profile(self, **kwargs):
        from db.models import CustomerProfile
        defaults = dict(user_id="u1", name="Test", customer_segment="regular")
        defaults.update(kwargs)
        return CustomerProfile(**defaults)

    def test_returns_offer_dict(self):
        profile = self._make_profile()
        client, q, _ = _supabase_ok()
        with patch("tools.offer_generator_tool.get_customer_profile", return_value=profile), \
             patch("tools.offer_generator_tool.get_client", return_value=client):
            from tools.offer_generator_tool import offer_generator_tool
            out = offer_generator_tool.invoke({
                "user_id": "u1", "session_id": "s1", "complaint_type": "damaged"
            })
        assert "offer_type" in out
        assert "offer_value" in out
        assert "offer_description" in out
        assert "risk_level" in out

    def test_no_profile_returns_default_offer(self):
        with patch("tools.offer_generator_tool.get_customer_profile", return_value=None):
            from tools.offer_generator_tool import offer_generator_tool
            out = offer_generator_tool.invoke({
                "user_id": "u_ghost", "session_id": "s1"
            })
        assert out["offer_type"] == "wallet_credit"
        assert out["risk_level"] == "unknown"

    def test_offer_value_respects_safety_cap(self):
        profile = self._make_profile(customer_segment="churning")
        client, q, _ = _supabase_ok()
        with patch("tools.offer_generator_tool.get_customer_profile", return_value=profile), \
             patch("tools.offer_generator_tool.get_client", return_value=client):
            from tools.offer_generator_tool import offer_generator_tool
            from config import settings
            out = offer_generator_tool.invoke({
                "user_id": "u1", "session_id": "s1"
            })
        if out["offer_type"] == "wallet_credit":
            assert out["offer_value"] <= settings.max_wallet_credit_inr
        elif out["offer_type"] == "discount":
            assert out["offer_value"] <= settings.max_discount_percent

    def test_db_persist_failure_does_not_raise(self):
        """DB write failure should be swallowed; offer still returned."""
        profile = self._make_profile()
        with patch("tools.offer_generator_tool.get_customer_profile", return_value=profile), \
             patch("tools.offer_generator_tool.get_client", side_effect=RuntimeError("DB down")):
            from tools.offer_generator_tool import offer_generator_tool
            out = offer_generator_tool.invoke({
                "user_id": "u1", "session_id": "s1"
            })
        assert "offer_type" in out

    def test_outer_exception_returns_failed(self):
        with patch("tools.offer_generator_tool.get_customer_profile", side_effect=Exception("boom")):
            from tools.offer_generator_tool import offer_generator_tool
            out = offer_generator_tool.invoke({"user_id": "u1", "session_id": "s1"})
        assert out["status"] == "failed"

    def test_risk_score_present_in_output(self):
        profile = self._make_profile()
        client, q, _ = _supabase_ok()
        with patch("tools.offer_generator_tool.get_customer_profile", return_value=profile), \
             patch("tools.offer_generator_tool.get_client", return_value=client):
            from tools.offer_generator_tool import offer_generator_tool
            out = offer_generator_tool.invoke({"user_id": "u1", "session_id": "s1"})
        assert "risk_score" in out
        assert 0.0 <= out["risk_score"] <= 1.0


# ===========================================================================
# dispatch_checklist_tool
# ===========================================================================

class TestDispatchChecklistTool:

    def test_returns_checklist_dict(self):
        client, q, _ = _supabase_ok()
        with patch("tools.dispatch_checklist_tool.get_client", return_value=client):
            from tools.dispatch_checklist_tool import dispatch_checklist_tool
            out = dispatch_checklist_tool.invoke({
                "user_id": "u1", "session_id": "s1",
                "issues": ["not_delivered"],
            })
        assert "checklist_items" in out
        assert "checklist_id" in out
        assert out["item_count"] == len(out["checklist_items"])

    def test_known_issue_returns_specific_items(self):
        client, q, _ = _supabase_ok()
        with patch("tools.dispatch_checklist_tool.get_client", return_value=client):
            from tools.dispatch_checklist_tool import dispatch_checklist_tool
            out = dispatch_checklist_tool.invoke({
                "user_id": "u1", "session_id": "s1",
                "issues": ["late_delivery"],
            })
        assert any("GPS" in item or "ETA" in item or "delay" in item.lower()
                   for item in out["checklist_items"])

    def test_multiple_issues_merged_deduped(self):
        client, q, _ = _supabase_ok()
        with patch("tools.dispatch_checklist_tool.get_client", return_value=client):
            from tools.dispatch_checklist_tool import dispatch_checklist_tool
            out = dispatch_checklist_tool.invoke({
                "user_id": "u1", "session_id": "s1",
                "issues": ["damaged_package", "missing_items"],
            })
        # Should have items from both issues, no duplicates
        assert len(out["checklist_items"]) == len(set(out["checklist_items"]))
        assert out["item_count"] > 4

    def test_unknown_issue_falls_back_to_default(self):
        client, q, _ = _supabase_ok()
        with patch("tools.dispatch_checklist_tool.get_client", return_value=client):
            from tools.dispatch_checklist_tool import dispatch_checklist_tool
            out = dispatch_checklist_tool.invoke({
                "user_id": "u1", "session_id": "s1",
                "issues": ["totally_unknown_issue_xyz"],
            })
        assert len(out["checklist_items"]) > 0
        # Default checklist has 4 items
        assert out["item_count"] == 4

    def test_empty_issues_falls_back_to_default(self):
        client, q, _ = _supabase_ok()
        with patch("tools.dispatch_checklist_tool.get_client", return_value=client):
            from tools.dispatch_checklist_tool import dispatch_checklist_tool
            out = dispatch_checklist_tool.invoke({
                "user_id": "u1", "session_id": "s1", "issues": [],
            })
        assert out["item_count"] > 0

    def test_checklist_id_unique(self):
        client, q, _ = _supabase_ok()
        with patch("tools.dispatch_checklist_tool.get_client", return_value=client):
            from tools.dispatch_checklist_tool import dispatch_checklist_tool
            r1 = dispatch_checklist_tool.invoke({"user_id": "u1", "session_id": "s1", "issues": []})
            r2 = dispatch_checklist_tool.invoke({"user_id": "u1", "session_id": "s1", "issues": []})
        assert r1["checklist_id"] != r2["checklist_id"]

    def test_db_error_does_not_raise(self):
        """DB write failure should be swallowed; checklist still returned."""
        with patch("tools.dispatch_checklist_tool.get_client", side_effect=RuntimeError("DB down")):
            from tools.dispatch_checklist_tool import dispatch_checklist_tool
            out = dispatch_checklist_tool.invoke({
                "user_id": "u1", "session_id": "s1",
                "issues": ["not_delivered"],
            })
        assert "checklist_items" in out

    def test_message_contains_count(self):
        client, q, _ = _supabase_ok()
        with patch("tools.dispatch_checklist_tool.get_client", return_value=client):
            from tools.dispatch_checklist_tool import dispatch_checklist_tool
            out = dispatch_checklist_tool.invoke({
                "user_id": "u1", "session_id": "s1",
                "issues": ["late_delivery"],
            })
        assert str(out["item_count"]) in out["message"]

    def test_all_known_issue_types_resolve(self):
        """Every key in _ISSUE_CHECKLIST must produce at least one item."""
        from tools.dispatch_checklist_tool import _ISSUE_CHECKLIST, dispatch_checklist_tool
        client, q, _ = _supabase_ok()
        with patch("tools.dispatch_checklist_tool.get_client", return_value=client):
            for key in _ISSUE_CHECKLIST:
                out = dispatch_checklist_tool.invoke({
                    "user_id": "u1", "session_id": "s1", "issues": [key],
                })
                assert out["item_count"] > 0, f"issue '{key}' produced empty checklist"


# ===========================================================================
# __init__.py — all tools importable
# ===========================================================================

class TestToolsInit:

    def test_all_tools_importable(self):
        from tools import (
            order_lookup_tool,
            refund_tool,
            replacement_tool,
            wallet_credit_tool,
            offer_generator_tool,
            dispatch_checklist_tool,
        )
        tools = [
            order_lookup_tool,
            refund_tool,
            replacement_tool,
            wallet_credit_tool,
            offer_generator_tool,
            dispatch_checklist_tool,
        ]
        for t in tools:
            assert hasattr(t, "name")
            assert hasattr(t, "invoke")

    def test_tool_names_match_module_names(self):
        from tools import (
            order_lookup_tool, refund_tool, replacement_tool,
            wallet_credit_tool, offer_generator_tool, dispatch_checklist_tool,
        )
        assert order_lookup_tool.name == "order_lookup_tool"
        assert refund_tool.name == "refund_tool"
        assert replacement_tool.name == "replacement_tool"
        assert wallet_credit_tool.name == "wallet_credit_tool"
        assert offer_generator_tool.name == "offer_generator_tool"
        assert dispatch_checklist_tool.name == "dispatch_checklist_tool"
