"""
backend/data/customer_sentiment.py

Appends customer sentiment data to a flat JSONL file for human analysis.
No API, no DB — humans decide what to do with this data.

Schema per line:
{
  "session_id": str,
  "user_id": str,
  "scenario": str,           # complaint_type or intent
  "resolution_type": str,    # refund | replacement | wallet_credit | human_handoff
  "csat_score": int,         # 5 = AI-resolved, lower = human-flagged
  "emotion_at_start": str,   # angry | neutral | upset | confused
  "language": str,
  "human_handoff": bool,
  "handoff_reason": str,     # why escalated (if human_handoff)
  "timestamp": str           # ISO UTC
}
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_SENTIMENT_FILE = Path(__file__).parent / "customer_sentiment.jsonl"


def log_sentiment(
    session_id: str,
    user_id: str,
    scenario: str,
    resolution_type: str,
    csat_score: int,
    emotion_at_start: str,
    language: str,
    human_handoff: bool = False,
    handoff_reason: str = "",
) -> None:
    """
    Append one sentiment record to customer_sentiment.jsonl.
    Fire-and-forget — never raises.
    """
    try:
        record = {
            "session_id": session_id,
            "user_id": user_id,
            "scenario": scenario,
            "resolution_type": resolution_type,
            "csat_score": csat_score,
            "emotion_at_start": emotion_at_start,
            "language": language,
            "human_handoff": human_handoff,
            "handoff_reason": handoff_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _SENTIMENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _SENTIMENT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.debug("Sentiment logged: session=%s csat=%d", session_id, csat_score)
    except Exception as exc:
        logger.warning("Sentiment logging failed: %s", exc)
