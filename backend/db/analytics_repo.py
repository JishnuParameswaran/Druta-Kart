"""
Druta Kart - Analytics repository.

Persists every interaction turn and provides aggregation queries for the
dashboard / observability layer.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from db.supabase_client import get_client

logger = logging.getLogger(__name__)

_MESSAGES_TABLE = "chat_messages"
_COMPLAINTS_TABLE = "complaint_logs"
_SESSIONS_TABLE = "chat_sessions"


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------

def log_interaction(
    session_id: str,
    user_id: str,
    message: str,
    response: str,
    emotion: Optional[str] = None,
    language: Optional[str] = None,
    intent: Optional[str] = None,
    agent: Optional[str] = None,
    tools: Optional[List[str]] = None,
    offer: Optional[str] = None,
    status: Optional[str] = None,
    latency: Optional[int] = None,
    tokens: Optional[int] = None,
    hallucination_flagged: bool = False,
) -> None:
    """Persist one turn (user message + bot response) to chat_messages.

    This is fire-and-forget: errors are logged but not re-raised so they
    never break the main request path.
    """
    try:
        client = get_client()
        now = datetime.utcnow().isoformat()

        rows = [
            {
                "session_id": session_id,
                "user_id": user_id,
                "role": "user",
                "content": message,
                "timestamp": now,
                "emotion": emotion,
                "language": language,
                "agent_used": agent,
                "tools_called": tools or [],
                "latency_ms": None,
                "tokens_used": None,
                "hallucination_flagged": False,
            },
            {
                "session_id": session_id,
                "user_id": user_id,
                "role": "bot",
                "content": response,
                "timestamp": now,
                "emotion": None,
                "language": language,
                "agent_used": agent,
                "tools_called": tools or [],
                "latency_ms": latency,
                "tokens_used": tokens,
                "hallucination_flagged": hallucination_flagged,
            },
        ]
        client.table(_MESSAGES_TABLE).insert(rows).execute()

        # Optionally update session status
        if status:
            client.table(_SESSIONS_TABLE).update(
                {"resolution_status": status}
            ).eq("session_id", session_id).execute()

    except Exception as exc:
        logger.error(
            "log_interaction failed for session=%s user=%s: %s",
            session_id, user_id, exc
        )


# ---------------------------------------------------------------------------
# Read / aggregation path
# ---------------------------------------------------------------------------

def get_complaint_analytics() -> Dict[str, Any]:
    """Return complaint aggregate metrics.

    Shape::

        {
            "total_complaints": int,
            "by_type": {"damaged": 3, "missing": 1, ...},
            "by_product": {"Tomatoes 500g": 2, ...},
            "resolution_rate": float,          # 0.0 – 1.0
            "avg_resolution_time": float,      # hours
        }
    """
    try:
        client = get_client()
        result = client.table(_COMPLAINTS_TABLE).select("*").execute()
        rows: List[Dict] = result.data or []

        total = len(rows)
        by_type: Dict[str, int] = {}
        by_product: Dict[str, int] = {}
        resolved_count = 0
        resolution_times: List[float] = []

        for row in rows:
            # by_type
            ctype = row.get("complaint_type", "unknown")
            by_type[ctype] = by_type.get(ctype, 0) + 1

            # by_product
            product = row.get("product_name") or "unknown"
            by_product[product] = by_product.get(product, 0) + 1

            # resolution
            rtype = row.get("resolution_type", "none")
            if rtype != "none":
                resolved_count += 1

            # avg resolution time
            created_raw = row.get("created_at")
            resolved_raw = row.get("resolved_at")
            if created_raw and resolved_raw:
                try:
                    created_dt = datetime.fromisoformat(str(created_raw))
                    resolved_dt = datetime.fromisoformat(str(resolved_raw))
                    diff_hours = (resolved_dt - created_dt).total_seconds() / 3600
                    if diff_hours >= 0:
                        resolution_times.append(diff_hours)
                except (ValueError, TypeError):
                    pass

        resolution_rate = round(resolved_count / total, 4) if total > 0 else 0.0
        avg_resolution_time = (
            round(sum(resolution_times) / len(resolution_times), 2)
            if resolution_times else 0.0
        )

        return {
            "total_complaints": total,
            "by_type": by_type,
            "by_product": by_product,
            "resolution_rate": resolution_rate,
            "avg_resolution_time": avg_resolution_time,
        }
    except Exception as exc:
        logger.error("get_complaint_analytics failed: %s", exc)
        raise


def get_customer_satisfaction_score() -> float:
    """Return the mean satisfaction score across all customers (0.0 – 5.0).

    Returns 0.0 if no scores are available.
    """
    try:
        client = get_client()
        result = (
            client.table("customers")
            .select("satisfaction_score")
            .not_.is_("satisfaction_score", "null")
            .execute()
        )
        rows: List[Dict] = result.data or []
        scores = [
            float(r["satisfaction_score"])
            for r in rows
            if r.get("satisfaction_score") is not None
        ]
        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 2)
    except Exception as exc:
        logger.error("get_customer_satisfaction_score failed: %s", exc)
        raise


def get_common_complaints(limit: int = 10) -> List[Dict[str, Any]]:
    """Return the most common complaint types ordered by frequency.

    Returns a list of dicts: [{"type": "damaged", "count": 42}, ...]
    """
    try:
        client = get_client()
        result = (
            client.table(_COMPLAINTS_TABLE)
            .select("complaint_type")
            .execute()
        )
        rows: List[Dict] = result.data or []

        counts: Dict[str, int] = {}
        for row in rows:
            ctype = row.get("complaint_type", "unknown")
            counts[ctype] = counts.get(ctype, 0) + 1

        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [
            {"type": ctype, "count": cnt}
            for ctype, cnt in sorted_counts[:limit]
        ]
    except Exception as exc:
        logger.error("get_common_complaints failed: %s", exc)
        raise


def get_daily_stats(days: int = 7) -> List[Dict[str, Any]]:
    """Return per-day interaction stats for the last *days* days.

    Returns a list ordered oldest → newest::

        [
            {
                "date": "2026-02-19",
                "total_messages": 120,
                "unique_users": 45,
                "complaints": 12,
                "hallucinations_flagged": 2,
            },
            ...
        ]
    """
    try:
        client = get_client()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        msg_result = (
            client.table(_MESSAGES_TABLE)
            .select("user_id, timestamp, hallucination_flagged, role")
            .gte("timestamp", cutoff)
            .execute()
        )
        complaint_result = (
            client.table(_COMPLAINTS_TABLE)
            .select("created_at")
            .gte("created_at", cutoff)
            .execute()
        )

        # Aggregate messages by date
        day_stats: Dict[str, Dict[str, Any]] = {}

        def _ensure_day(date_str: str) -> None:
            if date_str not in day_stats:
                day_stats[date_str] = {
                    "date": date_str,
                    "total_messages": 0,
                    "unique_users": set(),
                    "complaints": 0,
                    "hallucinations_flagged": 0,
                }

        for row in msg_result.data or []:
            ts = row.get("timestamp", "")
            try:
                date_str = str(ts)[:10]  # "YYYY-MM-DD"
                _ensure_day(date_str)
                day_stats[date_str]["total_messages"] += 1
                if row.get("user_id"):
                    day_stats[date_str]["unique_users"].add(row["user_id"])
                if row.get("hallucination_flagged"):
                    day_stats[date_str]["hallucinations_flagged"] += 1
            except (ValueError, TypeError):
                pass

        for row in complaint_result.data or []:
            ts = row.get("created_at", "")
            try:
                date_str = str(ts)[:10]
                _ensure_day(date_str)
                day_stats[date_str]["complaints"] += 1
            except (ValueError, TypeError):
                pass

        # Convert sets to counts, sort by date
        output = []
        for entry in sorted(day_stats.values(), key=lambda x: x["date"]):
            output.append({
                "date": entry["date"],
                "total_messages": entry["total_messages"],
                "unique_users": len(entry["unique_users"]),
                "complaints": entry["complaints"],
                "hallucinations_flagged": entry["hallucinations_flagged"],
            })

        return output
    except Exception as exc:
        logger.error("get_daily_stats failed: %s", exc)
        raise
