"""
Druta Kart - Dispatch checklist tool.

LangChain tool that maps reported delivery issues to actionable checklist
items for the dispatch team and persists the checklist to Supabase.
"""
import logging
from datetime import datetime
from typing import List
from uuid import uuid4

from langchain_core.tools import tool

from db.supabase_client import get_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Issue → checklist mapping
# ---------------------------------------------------------------------------

_ISSUE_CHECKLIST: dict[str, List[str]] = {
    "not_delivered": [
        "Verify delivery address with customer",
        "Check last GPS ping of delivery partner",
        "Confirm whether delivery was attempted",
        "Escalate to delivery partner supervisor if no response within 30 min",
    ],
    "wrong_address": [
        "Correct delivery address in the order management system",
        "Re-assign order to nearest available delivery partner",
        "Notify customer of corrected ETA",
    ],
    "late_delivery": [
        "Check current delivery partner location via GPS",
        "Identify delay cause (traffic / weather / partner unavailability)",
        "Provide customer with updated ETA",
        "Apply SLA-breach compensation if delay exceeds 60 min",
    ],
    "damaged_package": [
        "Instruct delivery partner not to hand over visibly damaged goods",
        "Initiate replacement order from nearest dark store",
        "Log damaged-package report with photo evidence",
        "Arrange pick-up of damaged goods within 24 h",
    ],
    "missing_items": [
        "Cross-check packing manifest against items reported missing",
        "Identify responsible packing station",
        "Initiate partial replacement for missing items",
        "Flag packing station for quality audit",
    ],
    "delivery_partner_unreachable": [
        "Attempt contact via secondary number",
        "Ping delivery partner app for last known location",
        "Re-assign order if no contact within 15 min",
        "File incident report for delivery partner management",
    ],
    "refused_delivery": [
        "Confirm reason for refusal with customer",
        "Check whether a re-delivery slot is required",
        "Update order status and notify warehouse",
        "Initiate return-to-origin process if customer does not want re-delivery",
    ],
}

_DEFAULT_CHECKLIST: List[str] = [
    "Review order details and full delivery history",
    "Contact delivery partner for status update",
    "Notify customer of ongoing investigation and ETA",
    "Escalate to dispatch supervisor if unresolved within 30 min",
]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool
def dispatch_checklist_tool(user_id: str, session_id: str, issues: List[str]) -> dict:
    """Generate and log a dispatch checklist for a delivery issue.

    Maps reported issue keywords to actionable steps for the dispatch team
    and writes the checklist to the database.

    Supported issue keywords: not_delivered, wrong_address, late_delivery,
    damaged_package, missing_items, delivery_partner_unreachable, refused_delivery.
    Unknown keywords fall back to a generic checklist.

    Args:
        user_id:    Customer user ID.
        session_id: Current chat session ID.
        issues:     List of issue keyword strings.

    Returns:
        Dict with keys: checklist_id, issues_reported, checklist_items,
        item_count, message.  On failure: status="failed" and 'error'.
    """
    try:
        # Build deduplicated checklist preserving insertion order
        checklist_items: List[str] = []
        seen: set[str] = set()
        matched_any = False

        for issue in issues:
            key = str(issue).strip().lower().replace(" ", "_")
            items = _ISSUE_CHECKLIST.get(key, [])
            if items:
                matched_any = True
            for item in items:
                if item not in seen:
                    checklist_items.append(item)
                    seen.add(item)

        if not matched_any:
            checklist_items = _DEFAULT_CHECKLIST.copy()

        checklist_id = str(uuid4())

        # Persist to DB (fire-and-forget — don't let DB error block the response)
        try:
            get_client().table("dispatch_checklists").insert({
                "checklist_id": checklist_id,
                "session_id": session_id,
                "user_id": user_id,
                "issues_reported": issues,
                "checklist_items": checklist_items,
                "sent_at": datetime.utcnow().isoformat(),
            }).execute()
        except Exception as db_exc:
            logger.warning("dispatch_checklist_tool: DB write failed: %s", db_exc)

        return {
            "checklist_id": checklist_id,
            "issues_reported": issues,
            "checklist_items": checklist_items,
            "item_count": len(checklist_items),
            "message": (
                f"Dispatch checklist generated with {len(checklist_items)} action items "
                f"and sent to the dispatch team."
            ),
        }
    except Exception as exc:
        logger.error("dispatch_checklist_tool failed for user=%s: %s", user_id, exc)
        return {"user_id": user_id, "status": "failed", "error": str(exc)}
