"""
Druta Kart - Supabase client singleton.

Usage:
    from db.supabase_client import get_client
    client = get_client()
    client.table("customers").select("*").execute()
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_client = None  # module-level singleton


def get_client():
    """Return the shared Supabase client, initialising it on first call.

    Raises RuntimeError if SUPABASE_URL or SUPABASE_ANON_KEY are missing so
    the rest of the application gets a clear error message rather than a
    cryptic AttributeError deep in the supabase-py stack.
    """
    global _client
    if _client is not None:
        return _client

    url: Optional[str] = os.getenv("SUPABASE_URL")
    key: Optional[str] = os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        missing = []
        if not url:
            missing.append("SUPABASE_URL")
        if not key:
            missing.append("SUPABASE_ANON_KEY")
        raise RuntimeError(
            f"Supabase client cannot be initialised: missing environment "
            f"variable(s): {', '.join(missing)}.  "
            f"Set them in your .env file or environment."
        )

    try:
        from supabase import create_client, Client  # type: ignore
        _client = create_client(url, key)
        logger.info("Supabase client initialised successfully.")
    except ImportError as exc:
        raise RuntimeError(
            "supabase-py is not installed.  Run: pip install supabase"
        ) from exc

    return _client


def reset_client() -> None:
    """Reset the singleton — useful in tests to inject a mock client."""
    global _client
    _client = None
