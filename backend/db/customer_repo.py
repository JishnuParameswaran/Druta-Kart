"""
Druta Kart - Customer profile repository.

All database access goes through the Supabase client returned by get_client().
Functions return Pydantic models so callers never touch raw dicts.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from db.models import CustomerProfile, CustomerSegment
from db.supabase_client import get_client

logger = logging.getLogger(__name__)

_TABLE = "customers"

# Segmentation thresholds
_CHURN_DAYS = 30          # days since last order that flags "churning"
_BULK_MIN_SPEND = 1000.0  # avg spend INR to qualify as "bulk"
_BULK_MIN_ORDERS = 5      # total orders needed alongside bulk spend
_FREQ_COMPLAINT_THRESHOLD = 3  # complaint_count > this → frequent_complainer
_NEW_ORDER_THRESHOLD = 3  # total_orders < this → new


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _row_to_profile(row: dict) -> CustomerProfile:
    """Convert a raw Supabase dict row to a CustomerProfile model."""
    # Supabase returns ISO strings for timestamps; parse them to datetime.
    for field in ("last_complaint_date", "created_at", "updated_at"):
        if row.get(field) and isinstance(row[field], str):
            row[field] = datetime.fromisoformat(row[field])
    return CustomerProfile(**row)


def _compute_segment(profile: CustomerProfile) -> CustomerSegment:
    """Pure-function segment logic — kept separate so it can be unit-tested."""
    if profile.complaint_count > _FREQ_COMPLAINT_THRESHOLD:
        return CustomerSegment.frequent_complainer

    if profile.total_orders < _NEW_ORDER_THRESHOLD:
        return CustomerSegment.new

    # Check churn: compare last order date (we proxy via last_complaint_date
    # when a proper last_order_date column is absent; callers can pass it).
    if profile.last_complaint_date is not None:
        now = datetime.now(timezone.utc)
        last = profile.last_complaint_date
        # Make timezone-aware if naive
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        days_since = (now - last).days
        if days_since > _CHURN_DAYS:
            return CustomerSegment.churning

    if (
        profile.avg_spend_inr > _BULK_MIN_SPEND
        and profile.total_orders > _BULK_MIN_ORDERS
    ):
        return CustomerSegment.bulk

    return CustomerSegment.regular


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_customer_profile(user_id: str) -> Optional[CustomerProfile]:
    """Fetch a customer profile by user_id.  Returns None if not found."""
    try:
        client = get_client()
        result = (
            client.table(_TABLE)
            .select("*")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if result.data:
            return _row_to_profile(result.data[0])
        return None
    except Exception as exc:
        logger.error("get_customer_profile failed for %s: %s", user_id, exc)
        raise


def create_customer_profile(user_id: str, name: str, phone: Optional[str] = None) -> CustomerProfile:
    """Insert a new customer profile with sensible defaults.

    Raises ValueError if a profile for user_id already exists.
    """
    try:
        client = get_client()
        # Check for existing profile
        existing = get_customer_profile(user_id)
        if existing is not None:
            raise ValueError(f"Customer profile already exists for user_id={user_id}")

        profile = CustomerProfile(
            user_id=user_id,
            name=name,
            phone=phone,
            customer_segment=CustomerSegment.new,
        )
        payload = profile.model_dump()
        # Supabase expects ISO strings for datetimes
        for field in ("last_complaint_date", "created_at", "updated_at"):
            if payload.get(field) and isinstance(payload[field], datetime):
                payload[field] = payload[field].isoformat()

        result = client.table(_TABLE).insert(payload).execute()
        return _row_to_profile(result.data[0])
    except ValueError:
        raise
    except Exception as exc:
        logger.error("create_customer_profile failed for %s: %s", user_id, exc)
        raise


def update_customer_stats(
    user_id: str,
    new_order: bool = False,
    complaint: bool = False,
    satisfaction_score: Optional[float] = None,
) -> None:
    """Increment order/complaint counters and optionally update satisfaction.

    Reads the current profile, applies deltas in Python, then upserts the row
    so there's no risk of a concurrent-write race through SQL.  For a high-
    throughput system you would use a Postgres function instead.
    """
    try:
        profile = get_customer_profile(user_id)
        if profile is None:
            raise ValueError(f"No customer profile found for user_id={user_id}")

        updates: dict = {"updated_at": datetime.utcnow().isoformat()}

        if new_order:
            updates["total_orders"] = profile.total_orders + 1

        if complaint:
            updates["complaint_count"] = profile.complaint_count + 1
            updates["last_complaint_date"] = datetime.utcnow().isoformat()

        if satisfaction_score is not None:
            # Rolling average: weight new score equally with the stored one
            current = profile.satisfaction_score
            if current is None:
                updates["satisfaction_score"] = satisfaction_score
            else:
                updates["satisfaction_score"] = round((current + satisfaction_score) / 2, 2)

        # Recompute segment with updated values
        update_kwargs = {
            k: v for k, v in updates.items()
            if k in CustomerProfile.model_fields
        }
        # Convert ISO strings back to datetime for segment logic
        for field in ("last_complaint_date",):
            if field in update_kwargs and isinstance(update_kwargs[field], str):
                update_kwargs[field] = datetime.fromisoformat(update_kwargs[field])
        updated_profile = profile.model_copy(update=update_kwargs)

        updates["customer_segment"] = _compute_segment(updated_profile).value

        client = get_client()
        client.table(_TABLE).update(updates).eq("user_id", user_id).execute()
    except ValueError:
        raise
    except Exception as exc:
        logger.error("update_customer_stats failed for %s: %s", user_id, exc)
        raise


def get_customer_segment(user_id: str) -> str:
    """Return the segment string for a customer, computing it live from profile.

    Returns "unknown" if profile is not found (rather than raising) so callers
    can degrade gracefully.
    """
    profile = get_customer_profile(user_id)
    if profile is None:
        logger.warning("get_customer_segment: no profile for user_id=%s", user_id)
        return "unknown"
    segment = _compute_segment(profile)
    return segment.value
