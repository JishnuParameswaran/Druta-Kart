"""
backend/data/synthetic/seed_orders.py

Generates 2-3 realistic fake orders per customer and inserts them into the
Supabase `orders` table.  Also saves a local orders_map.json so that the
AI-to-AI eval harness can pass real order IDs to the customer simulator.

Usage:
    cd backend
    python data/synthetic/seed_orders.py

Idempotent: uses upsert (on_conflict=order_id) so it is safe to re-run.
"""
from __future__ import annotations

import json
import os
import random
import string
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# ── Path setup ──────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).resolve().parent       # backend/data/synthetic/
BACKEND_DIR = SCRIPT_DIR.parents[1]                 # backend/
REPO_ROOT   = BACKEND_DIR.parent                    # repo root

sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

# ── File paths ───────────────────────────────────────────────────────────────

CONVERSATIONS_PATH = SCRIPT_DIR / "conversations.jsonl"
ORDERS_MAP_PATH    = SCRIPT_DIR / "orders_map.json"

# ── Static data pools ────────────────────────────────────────────────────────

_GROCERY_ITEMS = [
    "Amul Butter 500g",
    "Amul Gold Full Cream Milk 1L",
    "Tata Salt 1kg",
    "Aashirvaad Atta 5kg",
    "MDH Sambar Masala 100g",
    "MDH Rajma Masala 100g",
    "Fortune Sunflower Oil 1L",
    "Saffola Gold Oil 1L",
    "Parle-G Biscuits 800g",
    "Britannia Marie Gold 300g",
    "Haldiram's Aloo Bhujia 400g",
    "Haldiram's Namkeen Mix 400g",
    "Maggi 2-Minute Noodles 560g",
    "Kissan Mixed Fruit Jam 500g",
    "Nescafé Classic Coffee 50g",
    "Bru Instant Coffee 100g",
    "Tata Tea Gold 500g",
    "Lipton Yellow Label Tea 250g",
    "Lays Magic Masala 73g",
    "Kurkure Masala Munch 90g",
    "Cadbury Dairy Milk Silk 60g",
    "Kit Kat 4-Finger 41.5g",
    "Colgate Strong Teeth Toothpaste 300g",
    "Dove Beauty Soap 100g×4",
    "Surf Excel Easy Wash 1kg",
    "Vim Dishwash Bar 600g",
    "Dettol Liquid Handwash 200ml",
    "Tropicana Orange Juice 1L",
    "Real Mango Juice 1L",
    "Epigamia Greek Yogurt Mango 90g",
    "Mother Dairy Paneer 200g",
    "Fresho Cherry Tomatoes 200g",
    "Fresho Banana 1kg",
    "Paper Boat Aamras 250ml",
    "Red Bull Energy Drink 250ml",
]

_DELIVERY_PARTNERS = [
    "Blinkit",
    "Swiggy Instamart",
    "Zepto",
    "Dunzo",
    "BigBasket Now",
]

# Weighted: most orders are delivered; some still in transit or delayed
_STATUSES = [
    "delivered", "delivered", "delivered",
    "out_for_delivery",
    "delayed",
]

_BATCH_SIZE = 100   # rows per Supabase insert call


# ── Supabase REST client (bypasses supabase-py key-format validation) ────────

def _rest_headers() -> dict:
    key = os.environ["SUPABASE_ANON_KEY"]
    return {
        "apikey":        key,
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "Prefer":        "resolution=merge-duplicates,return=minimal",
    }


def _rest_url(table: str) -> str:
    base = os.environ["SUPABASE_URL"].rstrip("/")
    return f"{base}/rest/v1/{table}"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _random_order_id() -> str:
    chars = string.ascii_uppercase + string.digits
    suffix = "".join(random.choices(chars, k=8))
    return f"ORD-{suffix}"


def _make_order(user_id: str) -> dict:
    """Return a single fake order dict for the given user."""
    order_id = _random_order_id()
    status   = random.choice(_STATUSES)
    items    = random.sample(_GROCERY_ITEMS, random.randint(2, 5))
    amount   = round(random.uniform(120.0, 1500.0), 2)

    # estimated_delivery: 20-90 minutes from now (for in-transit) or in the past
    offset_mins = -random.randint(30, 2880) if status == "delivered" else random.randint(20, 90)
    est_delivery = datetime.now(timezone.utc) + timedelta(minutes=offset_mins)
    partner      = random.choice(_DELIVERY_PARTNERS)

    return {
        "order_id":           order_id,
        "user_id":            user_id,
        "status":             status,
        "items":              items,           # stored as JSONB array
        "amount_inr":         amount,
        "estimated_delivery": est_delivery.isoformat(),
        "delivery_partner":   partner,
        "tracking_url":       f"https://track.drutakart.in/{order_id}",
    }


def _insert_batch(batch: list[dict]) -> int:
    """Upsert a batch of order rows via Supabase REST API. Returns row count."""
    resp = requests.post(
        _rest_url("orders"),
        headers=_rest_headers(),
        json=batch,
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return len(batch)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Load all customer_ids from conversations.jsonl
    if not CONVERSATIONS_PATH.exists():
        sys.exit(f"conversations.jsonl not found at {CONVERSATIONS_PATH}")

    with CONVERSATIONS_PATH.open(encoding="utf-8") as f:
        conversations = [json.loads(line) for line in f if line.strip()]

    customer_ids = list({c["customer_id"] for c in conversations})
    print(f"Unique customers: {len(customer_ids)}")

    # 2. Generate 2-3 orders per customer
    orders_map: dict[str, list[dict]] = {}
    all_orders: list[dict] = []

    for cid in customer_ids:
        n = random.randint(2, 3)
        customer_orders = [_make_order(cid) for _ in range(n)]
        orders_map[cid] = customer_orders
        all_orders.extend(customer_orders)

    print(f"Orders generated: {len(all_orders)}")

    # 3. Insert into Supabase in batches via REST API
    total_inserted = 0
    batches = [all_orders[i:i + _BATCH_SIZE] for i in range(0, len(all_orders), _BATCH_SIZE)]

    for i, batch in enumerate(batches, 1):
        try:
            inserted = _insert_batch(batch)
            total_inserted += inserted
            print(f"  batch {i}/{len(batches)}: {inserted} rows upserted", flush=True)
        except Exception as exc:
            print(f"  batch {i}/{len(batches)}: ERROR — {exc}", flush=True)

    print(f"\nTotal rows upserted: {total_inserted}")

    # 4. Save orders_map.json (customer_id → list of order dicts)
    #    Eval script loads this to pass real order IDs to the simulator.
    ORDERS_MAP_PATH.write_text(
        json.dumps(orders_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"orders_map.json saved → {ORDERS_MAP_PATH}")

    # 5. Quick sanity check: fetch back one order
    sample_cid = customer_ids[0]
    sample_oid = orders_map[sample_cid][0]["order_id"]
    check = requests.get(
        _rest_url("orders"),
        headers={**_rest_headers(), "Prefer": ""},
        params={"order_id": f"eq.{sample_oid}", "select": "order_id,status,amount_inr,delivery_partner"},
        timeout=10,
    )
    if check.status_code == 200 and check.json():
        print(f"\nSanity check OK — fetched: {check.json()[0]}")
    else:
        print(f"\nSanity check FAILED (HTTP {check.status_code}) — order {sample_oid} not found in DB")


if __name__ == "__main__":
    main()
