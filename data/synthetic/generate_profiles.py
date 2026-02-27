#!/usr/bin/env python3
"""
Druta Kart — Synthetic Customer Profile Generator
Generates 3000 customer profiles saved to data/synthetic/profiles.jsonl

Usage:
    cd E:/Desktop/Projects/Druta-Kart
    python data/synthetic/generate_profiles.py

Requires GROQ_API_KEY in environment or .env file.
Groq is used only for generating realistic Indian names in batches.
All other fields are generated locally.
"""

import json
import os
import random
import sys
import uuid
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
OUTPUT_FILE = DATA_DIR / "synthetic" / "profiles.jsonl"

# ── ITEM → CATEGORY MAPPING ───────────────────────────────────────────────────
ITEM_TO_CATEGORY: dict[str, str] = {
    # Food / grocery
    "bread":       "damaged_food",
    "biscuit":     "damaged_food",
    "milk":        "damaged_food",
    "fruits":      "damaged_food",
    "vegetables":  "damaged_food",
    "rice":        "damaged_food",
    "dal":         "damaged_food",
    "chips":       "damaged_food",
    "namkeen":     "damaged_food",
    "cookies":     "damaged_food",
    "snacks":      "damaged_food",
    # Electronics
    "headphone":   "damaged_electronics",
    "charger":     "damaged_electronics",
    "phone":       "damaged_electronics",
    "laptop":      "damaged_electronics",
    "earbuds":     "damaged_electronics",
    "cable":       "damaged_electronics",
}
DEFAULT_CATEGORY = "damaged_food"

# ── IMAGE TYPE WEIGHTS (damaged_product complaints) ───────────────────────────
# (type, weight)
IMAGE_TYPE_WEIGHTS = [
    ("real_damage",       0.70),
    ("misidentification", 0.20),
    ("ai_generated",      0.10),
]
_IMG_TYPES  = [t for t, _ in IMAGE_TYPE_WEIGHTS]
_IMG_WEIGHTS = [w for _, w in IMAGE_TYPE_WEIGHTS]

IMAGE_TYPE_FOLDER: dict[str, str] = {
    "ai_generated":      "ai_generated_fake",
    "misidentification": "misidentification",
    # "real_damage" → category-based folder (damaged_food / damaged_electronics)
}

# ── DATA POOLS ────────────────────────────────────────────────────────────────
LANGUAGES = [
    "english", "hindi", "malayalam", "tamil",
    "kannada", "hinglish", "manglish", "kanglish",
]

FOOD_ITEMS = [
    "bread", "biscuit", "milk", "fruits", "vegetables", "rice", "dal",
    "chips", "namkeen", "cookies", "snacks", "atta", "sugar", "tea",
    "coffee", "ghee", "oil", "paneer", "curd", "butter", "eggs",
    "tomatoes", "onions", "potatoes", "bananas", "apples", "maggi",
    "poha", "upma mix", "sambhar powder", "rasam powder", "masala",
    "noodles", "pasta", "cornflakes", "oats", "honey", "jam",
    "chakki fresh atta", "basmati rice", "toor dal", "chana dal",
    "mustard oil", "coconut oil", "desi ghee", "amul butter",
    "alphonso mangoes", "mosambi juice", "tender coconut",
]

ELECTRONICS_ITEMS = [
    "headphone", "charger", "phone", "laptop", "earbuds", "cable",
    "USB hub", "mouse", "keyboard", "webcam", "power bank",
    "smart watch", "bluetooth speaker", "tablet cover", "screen guard",
    "OTG adapter", "HDMI cable", "Type-C cable", "fast charger",
    "wireless earbuds", "gaming mouse", "mechanical keyboard",
]

ALL_ITEMS = FOOD_ITEMS + ELECTRONICS_ITEMS

SCENARIO_ITEM_POOL: dict[str, list[str]] = {
    "damaged_product": FOOD_ITEMS[:20] + ELECTRONICS_ITEMS[:8],
    "late_delivery":   ALL_ITEMS,
    "wrong_item":      ALL_ITEMS,
    "missing_item":    ALL_ITEMS,
    "payment_issue":   [
        "wallet recharge", "UPI payment", "card payment",
        "COD order", "prepaid order", "refund credit",
    ],
    "happy_path":      ALL_ITEMS,
}

SCENARIO_EMOTIONS: dict[str, list[str]] = {
    "damaged_product": ["angry", "upset"],
    "late_delivery":   ["angry", "upset", "confused"],
    "wrong_item":      ["upset", "confused"],
    "missing_item":    ["upset", "angry"],
    "payment_issue":   ["confused", "upset", "angry"],
    "happy_path":      ["neutral"],
}

SCENARIO_RESOLUTIONS: dict[str, list[str]] = {
    "damaged_product": ["refund", "replacement", "wallet_credit"],
    "late_delivery":   ["tracking", "explanation", "wallet_credit"],
    "wrong_item":      ["replacement", "refund"],
    "missing_item":    ["replacement", "refund"],
    "payment_issue":   ["refund", "explanation"],
    "happy_path":      ["explanation"],
}

# ── SCENARIO DISTRIBUTION (must sum to 3000) ──────────────────────────────────
SCENARIO_DISTRIBUTION: dict[str, int] = {
    "damaged_product": 800,
    "late_delivery":   600,
    "wrong_item":      400,
    "missing_item":    400,
    "payment_issue":   400,
    "happy_path":      400,
}
TOTAL_PROFILES = sum(SCENARIO_DISTRIBUTION.values())  # 3000


# ── GROQ NAME GENERATION ──────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> list[dict]:
    """Strip markdown fences and parse JSON array."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        # parts[1] is the content inside the first fence pair
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    return json.loads(raw)


def generate_names_batch(client, count: int) -> list[dict]:
    """
    Call Groq to generate `count` realistic Indian customer names.
    Returns list of {"first_name": ..., "last_name": ..., "gender": ...}
    """
    prompt = (
        f"Generate exactly {count} realistic Indian customer names for an e-commerce app.\n"
        "Return ONLY a valid JSON array — no explanation, no markdown, no extra text.\n"
        "Each element must be an object with keys: first_name, last_name, gender (male or female).\n\n"
        "Use names from diverse Indian regions and communities:\n"
        "- North India: Sharma, Singh, Gupta, Verma, Yadav, Pandey families\n"
        "- South India: Iyer, Nair, Reddy, Pillai, Gowda, Naidu families\n"
        "- West India: Patel, Shah, Mehta, Desai families\n"
        "- East India: Mukherjee, Banerjee, Das, Roy families\n"
        "- Muslim names: Khan, Ansari, Shaikh, Siddiqui families\n"
        "- Sikh names: Singh, Kaur, Grewal, Sandhu families\n"
        "- Christian names: Fernandes, D'Souza, Rodrigues families\n"
        "Mix genders roughly 50/50. Make names sound authentic, not stereotypical."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.95,
        max_tokens=count * 25,
    )
    raw = response.choices[0].message.content
    names = _parse_json_response(raw)
    return names[:count]


def _fallback_name_pool(count: int) -> list[dict]:
    """Hardcoded Indian name pool used when Groq fails."""
    male_first = [
        "Arjun", "Rahul", "Vikram", "Suresh", "Ravi", "Manoj", "Pradeep", "Sanjay",
        "Rajesh", "Amit", "Deepak", "Ajay", "Vinay", "Naveen", "Arun", "Kartik",
        "Rohan", "Akash", "Nikhil", "Ankit", "Harsh", "Varun", "Mohit", "Pankaj",
        "Dinesh", "Ramesh", "Girish", "Mahesh", "Naresh", "Ganesh", "Sachin",
        "Srikanth", "Venkat", "Prasad", "Shiva", "Murali", "Balaji", "Senthil",
        "Krishnan", "Sudheer", "Imran", "Asif", "Farhan", "Khalid", "Danish",
        "Gurpreet", "Harpreet", "Jaswinder", "Manpreet", "Balvinder", "Kunal",
        "Siddharth", "Abhishek", "Vivek", "Gaurav", "Yogesh", "Hemant", "Ranjit",
    ]
    female_first = [
        "Priya", "Anjali", "Sunita", "Rekha", "Kavita", "Meena", "Anita", "Seema",
        "Pooja", "Neha", "Aarti", "Divya", "Geeta", "Shobha", "Usha", "Madhuri",
        "Deepa", "Sudha", "Radha", "Lalitha", "Saritha", "Kavitha", "Padma",
        "Lakshmi", "Saraswati", "Kamala", "Vimala", "Sumathi", "Revathi", "Geetha",
        "Nisha", "Ritu", "Sonia", "Manju", "Sangeeta", "Archana", "Vandana",
        "Fatima", "Ayesha", "Zara", "Rukhsar", "Shabana", "Nusrat", "Sadia",
        "Gurjeet", "Navneet", "Simran", "Kirandeep", "Amandeep", "Tanvi",
        "Shreya", "Aditi", "Pallavi", "Swati", "Rashmi", "Sneha", "Aparna",
    ]
    last_names = [
        "Kumar", "Sharma", "Singh", "Verma", "Gupta", "Patel", "Shah", "Joshi",
        "Mehta", "Agarwal", "Mishra", "Yadav", "Pandey", "Tiwari", "Dubey",
        "Reddy", "Naidu", "Rao", "Pillai", "Nair", "Menon", "Iyer", "Iyengar",
        "Krishnamurthy", "Subramaniam", "Venkataraman", "Balasubramaniam",
        "Mukherjee", "Chatterjee", "Banerjee", "Das", "Roy", "Sen",
        "Desai", "Parekh", "Modi", "Trivedi", "Bhatt", "Rawal",
        "Khan", "Ansari", "Shaikh", "Siddiqui", "Qureshi", "Malik",
        "Kaur", "Sandhu", "Grewal", "Dhaliwal", "Sidhu", "Brar",
        "Fernandes", "D'Souza", "Mascarenhas", "Rodrigues",
        "Hegde", "Shetty", "Gowda", "Naik", "Kamath",
    ]
    result = []
    for _ in range(count):
        gender = random.choice(["male", "female"])
        first = random.choice(male_first if gender == "male" else female_first)
        last = random.choice(last_names)
        result.append({"first_name": first, "last_name": last, "gender": gender})
    return result


def fetch_all_names(groq_api_key: str, total: int, batch_size: int = 100) -> list[dict]:
    """Fetch all names from Groq in batches, falling back to local pool on error."""
    try:
        from groq import Groq
    except ImportError:
        print("WARNING: groq package not installed. Using fallback names.")
        return _fallback_name_pool(total)

    client = Groq(api_key=groq_api_key)
    all_names: list[dict] = []

    full_batches, remainder = divmod(total, batch_size)
    total_batches = full_batches + (1 if remainder else 0)

    print(f"Generating {total:,} Indian names via Groq "
          f"({total_batches} batch{'es' if total_batches > 1 else ''} of up to {batch_size})…")

    for i in range(full_batches):
        print(f"  [{i+1:>3}/{total_batches}] batch of {batch_size}…", end=" ", flush=True)
        try:
            batch = generate_names_batch(client, batch_size)
            if len(batch) < batch_size:
                batch += _fallback_name_pool(batch_size - len(batch))
            all_names.extend(batch[:batch_size])
            print("OK")
        except Exception as exc:
            print(f"FAILED ({exc}) — fallback")
            all_names.extend(_fallback_name_pool(batch_size))

    if remainder:
        print(f"  [{total_batches:>3}/{total_batches}] batch of {remainder}…", end=" ", flush=True)
        try:
            batch = generate_names_batch(client, remainder)
            if len(batch) < remainder:
                batch += _fallback_name_pool(remainder - len(batch))
            all_names.extend(batch[:remainder])
            print("OK")
        except Exception as exc:
            print(f"FAILED ({exc}) — fallback")
            all_names.extend(_fallback_name_pool(remainder))

    return all_names[:total]


# ── IMAGE RESOLUTION ──────────────────────────────────────────────────────────
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

def _list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [f for f in folder.iterdir() if f.suffix.lower() in _IMAGE_EXT]


def resolve_image(complaint_item: str) -> dict:
    """
    Determine image_path and metadata for a damaged_product complaint.

    Returns a dict with keys:
        image_path      – relative path from repo root (forward slashes)
        image_type      – real_damage | misidentification | ai_generated
        image_category  – damaged_food | damaged_electronics
        image_needed    – always True for damaged_product
        needs_image     – True if placeholder (folder was empty)
    """
    # Determine category from item name
    item_lower = complaint_item.lower()
    category = DEFAULT_CATEGORY
    for keyword, cat in ITEM_TO_CATEGORY.items():
        if keyword in item_lower:
            category = cat
            break

    # Choose image type
    image_type: str = random.choices(_IMG_TYPES, weights=_IMG_WEIGHTS, k=1)[0]

    # Resolve folder
    if image_type == "real_damage":
        folder = IMAGES_DIR / category
    else:
        folder = IMAGES_DIR / IMAGE_TYPE_FOLDER[image_type]

    files = _list_images(folder)

    if files:
        chosen = random.choice(files)
        rel_path = chosen.relative_to(REPO_ROOT).as_posix()
        return {
            "image_path":     rel_path,
            "image_type":     image_type,
            "image_category": category,
            "image_needed":   True,
            "needs_image":    False,
        }
    else:
        # Folder missing or empty — store a descriptive placeholder
        placeholder = f"data/images/{folder.name}/placeholder.jpg"
        return {
            "image_path":     placeholder,
            "image_type":     image_type,
            "image_category": category,
            "image_needed":   True,
            "needs_image":    True,  # caller should populate image later
        }


# ── SEGMENT COMPUTATION ───────────────────────────────────────────────────────
def compute_segment(total_orders: int, avg_spend_inr: float, is_churning: bool) -> str:
    if total_orders < 3:
        return "new"
    if total_orders > 50:
        return "vip"
    if avg_spend_inr > 1000 and total_orders > 5:
        return "bulk"
    if is_churning:
        return "churning"
    return "regular"


# ── PROFILE BUILDER ───────────────────────────────────────────────────────────
def _make_scenario_sequence() -> list[str]:
    """Build a shuffled list of 3000 scenarios matching SCENARIO_DISTRIBUTION."""
    seq: list[str] = []
    for scenario, count in SCENARIO_DISTRIBUTION.items():
        seq.extend([scenario] * count)
    random.shuffle(seq)
    return seq


def generate_profiles(names: list[dict]) -> list[dict]:
    scenarios = _make_scenario_sequence()
    profiles: list[dict] = []

    for name_data, scenario in zip(names, scenarios):
        total_orders   = random.randint(1, 100)
        avg_spend_inr  = round(random.uniform(50.0, 5000.0), 2)
        is_churning    = random.random() < 0.15  # ~15% marked churning

        segment = compute_segment(total_orders, avg_spend_inr, is_churning)

        purchase_history = random.choices(ALL_ITEMS, k=random.randint(5, 15))
        complaint_item   = random.choice(SCENARIO_ITEM_POOL[scenario])
        emotion          = random.choice(SCENARIO_EMOTIONS[scenario])
        resolution       = random.choice(SCENARIO_RESOLUTIONS[scenario])

        profile: dict = {
            "customer_id":         str(uuid.uuid4()),
            "customer_name":       f"{name_data['first_name']} {name_data['last_name']}",
            "gender":              name_data["gender"],
            "age":                 random.randint(18, 65),
            "language":            random.choice(LANGUAGES),
            "total_orders":        total_orders,
            "avg_spend_inr":       avg_spend_inr,
            "purchase_history":    purchase_history,
            "complaint_scenario":  scenario,
            "complaint_item":      complaint_item,
            "expected_emotion":    emotion,
            "expected_resolution": resolution,
            "customer_segment":    segment,
        }

        # Image fields
        if scenario == "damaged_product":
            profile.update(resolve_image(complaint_item))
        else:
            profile["image_path"]     = None
            profile["image_type"]     = None
            profile["image_category"] = None
            profile["image_needed"]   = False
            profile["needs_image"]    = False

        profiles.append(profile)

    return profiles


# ── STATS PRINTER ─────────────────────────────────────────────────────────────
def print_stats(profiles: list[dict]) -> None:
    from collections import Counter

    sc  = Counter(p["complaint_scenario"]  for p in profiles)
    seg = Counter(p["customer_segment"]    for p in profiles)
    lang = Counter(p["language"]           for p in profiles)
    needs_img = sum(1 for p in profiles if p.get("needs_image"))

    print("\n── Scenario distribution ─────────────────────")
    for k, v in sorted(sc.items()):
        print(f"  {k:<22} {v:>5}")

    print("\n── Customer segments ─────────────────────────")
    for k, v in sorted(seg.items()):
        print(f"  {k:<12} {v:>5}")

    print("\n── Languages ─────────────────────────────────")
    for k, v in sorted(lang.items()):
        print(f"  {k:<12} {v:>5}")

    if needs_img:
        print(f"\n── needs_image=True (placeholder paths): {needs_img}")
        print("   Add real images to data/images/ folders and re-run to resolve.")
    else:
        print("\n── All image paths resolved from existing files.")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main() -> None:
    random.seed(42)  # reproducible shuffle; remove for fresh randomness each run

    # Resolve GROQ_API_KEY
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GROQ_API_KEY=") and not line.startswith("#"):
                    groq_api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if not groq_api_key:
        print(
            "ERROR: GROQ_API_KEY not found.\n"
            "Set it via environment variable or add it to .env at the project root."
        )
        sys.exit(1)

    # Ensure output dir exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate names via Groq
    names = fetch_all_names(groq_api_key, TOTAL_PROFILES, batch_size=100)

    # Pad with fallbacks if we got fewer than expected
    if len(names) < TOTAL_PROFILES:
        shortfall = TOTAL_PROFILES - len(names)
        print(f"Padding {shortfall} missing names with fallback pool…")
        names.extend(_fallback_name_pool(shortfall))
    names = names[:TOTAL_PROFILES]

    # Step 2: Build profiles
    print(f"\nBuilding {TOTAL_PROFILES:,} customer profiles…")
    profiles = generate_profiles(names)

    # Step 3: Write JSONL
    print(f"Writing to {OUTPUT_FILE}…")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        for profile in profiles:
            fh.write(json.dumps(profile, ensure_ascii=False) + "\n")

    print(f"\nDone — {len(profiles):,} profiles written to {OUTPUT_FILE.relative_to(REPO_ROOT)}")
    print_stats(profiles)


if __name__ == "__main__":
    main()
