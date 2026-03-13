"""
backend/security/safety_layer.py

Input → Processing → Output safety pipeline for Druta Kart.

Pipeline:
  run_input_safety()  → sanitize_input → check_prompt_injection → check_red_team
  run_output_safety() → moderate_output → mask_pii_in_logs → check_system_prompt_protection
"""
from __future__ import annotations

import re
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ── Regex patterns ─────────────────────────────────────────────────────────────

_SCRIPT_RE = re.compile(r"<script[\s\S]*?</script>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>", re.IGNORECASE)

_SQL_RE = re.compile(
    r"\b(DROP|SELECT|INSERT|UPDATE|DELETE|UNION|TRUNCATE|ALTER|CREATE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)

# 5+ consecutive chars that aren't: alphanumeric, whitespace, Indic scripts (U+0900–U+0D7F),
# or common chat punctuation .,!?@#₹%&*()-_+=:;'"/\
_EXCESS_SPECIAL_RE = re.compile(
    r"[^\w\s\u0900-\u0D7F.,!?@#₹%&*()\-_+=:;'\"/\\]{5,}",
    re.UNICODE,
)

# Indian mobile (primary): optional +91 prefix, 6–9 leading digit, 9 more digits
# Generic fallback: +CC (area) local
_PHONE_RE = re.compile(
    r"(?<!\d)(\+?91[\s\-]?)?[6-9]\d{9}(?!\d)"
    r"|(?<!\d)(\+?[0-9]{1,3}[\s\-]?)?\(?\d{3,5}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}(?!\d)",
)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)

# ── Threat phrase lists ────────────────────────────────────────────────────────

_INJECTION_PHRASES: list[str] = [
    "ignore all instructions",
    "ignore previous instructions",
    "ignore all previous",          # covers "ignore all previous instructions"
    "ignore your instructions",     # covers "ignore all your instructions"
    "ignore all your",              # covers "ignore all your instructions/rules/etc"
    "disregard instructions",
    "forget your instructions",
    "you are now",
    "pretend you are",
    "act as if",
    "jailbreak",
    "bypass",
    "override instructions",
    "dan mode",
    "do anything now",
    "bypass all restrictions",
    "bypass your safety",
]

_RED_TEAM_PHRASES: list[str] = [
    "show system prompt",
    "show me your system",       # covers "show me your system prompt"
    "reveal your system",
    "what are your instructions",
    "give me admin access",
    "show other users",
    "reveal your prompt",
    "what tools do you have",
    "what tools you have",       # covers "what tools you have available"
    "list your tools",
    "show all customer data",
    "i am the admin",
    "i am admin",
]

# SQL injection attack patterns — checked on ORIGINAL text before sanitization
_SQL_INJECTION_RE = re.compile(
    r"'[\s;]*(?:DROP|DELETE|INSERT|UPDATE|CREATE|ALTER|TRUNCATE|EXEC)\s"
    r"|UNION\s+(?:ALL\s+)?SELECT\s"
    r"|--\s*$"
    r"|\bOR\s+'?1'?\s*=\s*'?1'?"
    r"|/\*.*?\*/",
    re.IGNORECASE | re.MULTILINE,
)

# Human-readable alert messages for each blocked_reason
_SECURITY_ALERT_MESSAGES: dict[str, str] = {
    "prompt_injection": (
        "⚠️ Security Alert: Prompt injection attempt detected. "
        "This session has been flagged and escalated for human review."
    ),
    "red_team_probe": (
        "⚠️ Security Alert: System probe attempt detected. "
        "This session has been flagged and escalated for human review."
    ),
    "sql_injection": (
        "⚠️ Security Alert: SQL injection pattern detected. "
        "This session has been flagged and escalated for human review."
    ),
}

# Fragments that should never appear in an outbound response
_SYSTEM_PROMPT_MARKERS: list[str] = [
    "you are a customer support agent",
    "druta kart support",
    "your role is to",
    "system:",
    "you are an ai assistant",
    "<system>",
    "[system]",
    "assistant:",
]


# ── 1. sanitize_input ─────────────────────────────────────────────────────────

def sanitize_input(text: str) -> str:
    """Strip script/HTML tags, SQL injection keywords, and excessive special chars."""
    text = _SCRIPT_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _SQL_RE.sub("", text)
    text = _EXCESS_SPECIAL_RE.sub(" ", text)
    text = re.sub(r" {2,}", " ", text).strip()
    return text


# ── 2. check_prompt_injection ─────────────────────────────────────────────────

def check_prompt_injection(text: str) -> bool:
    """Return True if text contains a known prompt-injection phrase."""
    lower = text.lower()
    return any(phrase in lower for phrase in _INJECTION_PHRASES)


# ── 3. check_red_team ─────────────────────────────────────────────────────────

def check_red_team(text: str) -> bool:
    """Return True if text is probing for system internals or other users' data."""
    lower = text.lower()
    return any(phrase in lower for phrase in _RED_TEAM_PHRASES)


def check_sql_injection(text: str) -> bool:
    """Return True if text contains SQL injection attack patterns.

    Checked against the ORIGINAL (pre-sanitisation) text so that stripping
    SQL keywords doesn't erase the evidence before the check runs.
    """
    return bool(_SQL_INJECTION_RE.search(text))


def get_security_alert_message(blocked_reason: str | None) -> str:
    """Return the human-readable ⚠️ security alert for a given blocked_reason."""
    return _SECURITY_ALERT_MESSAGES.get(
        blocked_reason or "",
        (
            "⚠️ Security Alert: Suspicious input detected. "
            "This session has been flagged and escalated for human review."
        ),
    )


# ── 4. check_session_context ──────────────────────────────────────────────────

def check_session_context(
    user_id: str,
    session_id: str,
    data: dict[str, Any],
) -> bool:
    """
    Verify that any customer_id embedded in *data* matches *user_id*.
    Returns False (and logs a warning) if a mismatch is detected.

    Note: order-level ownership checks require a DB round-trip and should be
    done in the agent/tool layer where a Supabase client is already available.
    """
    claimed_cid = data.get("customer_id")
    if claimed_cid and str(claimed_cid) != str(user_id):
        log.warning(
            "session_context_mismatch",
            user_id=user_id,
            session_id=session_id,
            claimed_customer_id=claimed_cid,
        )
        return False
    return True


# ── 5. check_system_prompt_protection ────────────────────────────────────────

def check_system_prompt_protection(text: str) -> str:
    """
    Remove any line from *text* that resembles leaked system-prompt content.
    Returns cleaned text.
    """
    lines = text.splitlines()
    safe_lines = [
        line for line in lines
        if not any(marker in line.lower() for marker in _SYSTEM_PROMPT_MARKERS)
    ]
    return "\n".join(safe_lines).strip()


# ── 6. moderate_output ────────────────────────────────────────────────────────

def moderate_output(response_text: str, current_user_id: str) -> str:
    """
    Scrub other users' UUIDs from *response_text*.
    The current user's own UUID is left untouched.
    Phones and emails are handled by mask_pii_in_logs downstream.
    """
    def _check_uuid(m: re.Match) -> str:
        uid = m.group(0)
        return uid if uid.lower() == current_user_id.lower() else "[USER REDACTED]"

    return _UUID_RE.sub(_check_uuid, response_text)


# ── 7. mask_pii_in_logs ───────────────────────────────────────────────────────

def mask_pii_in_logs(text: str) -> str:
    """
    Partially mask PII for safe output and log strings:
      - Phone  →  last 4 digits only:  ******1234
      - Email  →  first 2 chars + @domain:  jo***@example.com
    """
    def _mask_phone(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) < 4:
            return "****"
        return "*" * (len(digits) - 4) + digits[-4:]

    def _mask_email(m: re.Match) -> str:
        addr = m.group(0)
        local, _, domain = addr.partition("@")
        masked = local[:2] + "*" * max(len(local) - 2, 0)
        return f"{masked}@{domain}"

    text = _PHONE_RE.sub(_mask_phone, text)
    text = _EMAIL_RE.sub(_mask_email, text)
    return text


# ── 8. log_security_event ─────────────────────────────────────────────────────

def log_security_event(
    event_type: str,
    user_id: str | None,
    session_id: str | None,
    detail: str,
) -> None:
    """
    Write a security event to structlog and Supabase security_events table.
    Fire-and-forget — never raises so the hot path is never blocked.
    """
    log.warning(
        "security_event",
        event_type=event_type,
        user_id=user_id,
        session_id=session_id,
        detail=detail,
    )
    try:
        from db.supabase_client import get_client
        get_client().table("security_events").insert({
            "event_type": event_type,
            "user_id": user_id,
            "session_id": session_id,
            "detail": detail[:1000],   # guard against oversized rows
        }).execute()
    except Exception as exc:
        log.debug("security_event_supabase_write_failed", error=str(exc))


# ── 9. run_input_safety ───────────────────────────────────────────────────────

def run_input_safety(
    text: str,
    user_id: str,
    session_id: str,
) -> dict[str, Any]:
    """
    Full input safety pipeline:
      sql_injection (original) → sanitize → injection check → red-team check.

    Returns:
        {
            "safe":           bool,
            "cleaned_text":   str,
            "blocked_reason": str | None
              # "sql_injection" | "prompt_injection" | "red_team_probe" | None
        }
    """
    # SQL injection must be checked on the ORIGINAL text before sanitize strips the keywords
    if check_sql_injection(text):
        log_security_event("sql_injection", user_id, session_id, detail=text[:200])
        return {"safe": False, "cleaned_text": text, "blocked_reason": "sql_injection"}

    cleaned = sanitize_input(text)

    if check_prompt_injection(cleaned):
        log_security_event("prompt_injection", user_id, session_id, detail=cleaned[:200])
        return {"safe": False, "cleaned_text": cleaned, "blocked_reason": "prompt_injection"}

    if check_red_team(cleaned):
        log_security_event("red_team_probe", user_id, session_id, detail=cleaned[:200])
        return {"safe": False, "cleaned_text": cleaned, "blocked_reason": "red_team_probe"}

    return {"safe": True, "cleaned_text": cleaned, "blocked_reason": None}


# ── 10. run_output_safety ─────────────────────────────────────────────────────

def run_output_safety(
    text: str,
    user_id: str,
    session_id: str,
) -> str:
    """
    Full output safety pipeline:
      moderate_output (UUID scrub) → mask_pii_in_logs (phone/email) → system-prompt strip.

    Returns clean response string.
    """
    text = moderate_output(text, user_id)
    text = mask_pii_in_logs(text)
    text = check_system_prompt_protection(text)
    return text
