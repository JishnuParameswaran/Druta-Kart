#!/usr/bin/env python3
"""
Druta Kart — Synthetic Conversation Dataset Generator

Generates 500 synthetic customer support conversations using Groq API.
Each conversation is a 3-5 exchange (6-10 message) dialogue between a customer
and support agent, in the customer's language.

Output: backend/data/synthetic/conversations.jsonl

Usage:
    cd E:/Desktop/Projects/Druta-Kart
    python backend/data/synthetic/generate_dataset.py

Requires GROQ_API_KEY in environment or .env file.
Profiles must exist at backend/data/synthetic/profiles.jsonl
(or fallback: data/synthetic/profiles.jsonl).
"""

import json
import os
import sys
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent        # backend/data/synthetic/
REPO_ROOT  = SCRIPT_DIR.parents[2]                  # project root (3 dirs up)

# Profiles: prefer sibling file; fall back to root-level data/synthetic/
_PROFILES_PRIMARY  = SCRIPT_DIR / "profiles.jsonl"
_PROFILES_FALLBACK = REPO_ROOT / "data" / "synthetic" / "profiles.jsonl"
PROFILES_FILE = _PROFILES_PRIMARY if _PROFILES_PRIMARY.exists() else _PROFILES_FALLBACK

OUTPUT_FILE = SCRIPT_DIR / "conversations.jsonl"

# ── Generation config ──────────────────────────────────────────────────────────
TARGET_COUNT  = 500
BATCH_SIZE    = 10
BATCH_SLEEP   = 5.0    # seconds between batches (Groq rate-limit courtesy)
SAVE_INTERVAL = 50     # flush progress to disk every N conversations
GROQ_MODEL    = "llama-3.3-70b-versatile"

# ── Scenario → intent mapping ──────────────────────────────────────────────────
SCENARIO_TO_INTENT: dict[str, str] = {
    "damaged_product": "complaint_damaged_item",
    "late_delivery":   "complaint_late_delivery",
    "wrong_item":      "complaint_wrong_item",
    "missing_item":    "complaint_missing_item",
    "payment_issue":   "inquiry_payment",
    "happy_path":      "general_inquiry",
}

SCENARIO_DESCRIPTION: dict[str, str] = {
    "damaged_product": "Customer received a damaged {item} and wants a refund, replacement, or wallet credit",
    "late_delivery":   "Customer's {item} order is delayed or has not arrived and they want an update",
    "wrong_item":      "Customer received a wrong item instead of the {item} they ordered",
    "missing_item":    "Customer's {item} is missing from their delivered order",
    "payment_issue":   "Customer has a payment or refund issue related to a {item} purchase",
    "happy_path":      "Customer is satisfied and asking a general question about their {item} order",
}

# Scenario weights matching the 3000-profile distribution
SCENARIO_WEIGHTS: dict[str, int] = {
    "damaged_product": 800,
    "late_delivery":   600,
    "wrong_item":      400,
    "missing_item":    400,
    "payment_issue":   400,
    "happy_path":      400,
}

# Language → LLM instruction
LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "english":   "English",
    "hindi":     "Hindi using Devanagari script (हिंदी)",
    "tamil":     "Tamil using Tamil script (தமிழ்)",
    "kannada":   "Kannada using Kannada script (ಕನ್ನಡ)",
    "malayalam": "Malayalam using Malayalam script (മലയാളം)",
    "marathi":   "Marathi using Devanagari script (मराठी)",
    "hinglish":  "Hinglish — Roman-script mix of Hindi and English (e.g. 'mera order abhi nahi aaya yaar')",
    "manglish":  "Manglish — Roman-script mix of Malayalam and English (e.g. 'entey order enga poyi')",
    "kanglish":  "Kanglish — Roman-script mix of Kannada and English (e.g. 'nanna order yaav side ide')",
}

# ── Prompt template ────────────────────────────────────────────────────────────
_PROMPT = """\
Generate a realistic quick-commerce customer support chat for "Druta Kart" (Indian grocery/electronics delivery app).

Customer details:
- Name: {name}
- Situation: {scenario_desc}
- Emotional tone: {emotion}
- Customer tier: {customer_segment}
- Language: {language_instruction}

Generate EXACTLY {num_turns} total messages in strict alternation starting with the customer.
Pattern: customer → agent → customer → agent → ...  (the last message must be from the agent).

Rules:
1. BOTH parties speak exclusively in {language_instruction} — no switching.
2. Customer expresses the situation with a {emotion} tone naturally.
3. Agent is empathetic, professional, and moves toward this resolution: {resolution}.
4. Write short chat-style messages (1–3 sentences each), not formal letters.
5. Agent verifies the order/issue briefly, then offers a concrete action.
6. For Indic scripts (Hindi, Tamil, Kannada, Malayalam, Marathi): use the native script, not Roman transliteration.

Return ONLY a JSON object — no markdown, no explanation:
{{
  "turns": [
    {{"role": "customer", "content": "..."}},
    {{"role": "agent",    "content": "..."}},
    {{"role": "customer", "content": "..."}},
    {{"role": "agent",    "content": "..."}}
  ]
}}\
"""

# ── Profile loading ────────────────────────────────────────────────────────────

def load_profiles() -> list[dict]:
    if not PROFILES_FILE.exists():
        print(
            f"ERROR: Profiles file not found.\n"
            f"  Checked: {_PROFILES_PRIMARY}\n"
            f"  Checked: {_PROFILES_FALLBACK}\n"
            "Run generate_profiles.py first to create profiles."
        )
        sys.exit(1)

    profiles: list[dict] = []
    with open(PROFILES_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                profiles.append(json.loads(line))

    print(f"Loaded {len(profiles):,} profiles from {PROFILES_FILE.relative_to(REPO_ROOT)}")
    return profiles


def sample_proportionally(profiles: list[dict], n: int) -> list[dict]:
    """
    Sample n profiles covering all scenarios proportionally to their weights.
    Groups by scenario, calculates target count per group, samples without
    replacement within each group, then shuffles the combined result.
    """
    by_scenario: dict[str, list[dict]] = {}
    for p in profiles:
        sc = p.get("complaint_scenario", "happy_path")
        by_scenario.setdefault(sc, []).append(p)

    total_weight = sum(SCENARIO_WEIGHTS.get(sc, 1) for sc in by_scenario)
    selected: list[dict] = []

    for sc, pool in by_scenario.items():
        weight = SCENARIO_WEIGHTS.get(sc, 1)
        target = round(n * weight / total_weight)
        draw   = min(target, len(pool))
        selected.extend(random.sample(pool, draw))

    # Trim or top-up to exactly n
    random.shuffle(selected)
    if len(selected) > n:
        selected = selected[:n]
    elif len(selected) < n:
        selected.extend(random.choices(profiles, k=n - len(selected)))

    random.shuffle(selected)
    return selected


# ── Conversation generation ────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Remove ``` code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    return text


def generate_conversation(client, profile: dict) -> list[dict]:
    """
    Call Groq to produce a multi-turn conversation for the given profile.
    Returns a list of {"role": ..., "content": ...} dicts.
    Retries up to 3 times on parse/structure errors.
    """
    scenario   = profile.get("complaint_scenario", "happy_path")
    item       = profile.get("complaint_item", "order")
    language   = profile.get("language", "english")
    emotion    = profile.get("expected_emotion", "neutral")
    segment    = profile.get("customer_segment", "regular")
    resolution = profile.get("expected_resolution", "explanation")
    name       = profile.get("customer_name", "Customer")

    lang_instr    = LANGUAGE_INSTRUCTIONS.get(language, "English")
    scenario_desc = SCENARIO_DESCRIPTION[scenario].format(item=item)
    num_pairs     = random.randint(3, 5)           # 3–5 exchanges
    num_turns     = num_pairs * 2                  # always even; ends with agent

    prompt = _PROMPT.format(
        name=name,
        scenario_desc=scenario_desc,
        emotion=emotion,
        customer_segment=segment,
        language_instruction=lang_instr,
        num_turns=num_turns,
        resolution=resolution,
    )

    customer_id = profile.get("customer_id", "unknown")
    last_exc: Exception | None = None
    raw: str = ""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            raw   = resp.choices[0].message.content
            data  = json.loads(_strip_markdown(raw))
            turns = data.get("turns", [])
            if not turns:
                raise ValueError("Empty turns list returned")
            for t in turns:
                if "role" not in t or "content" not in t:
                    raise ValueError(f"Malformed turn: {t}")
            return turns
        except Exception as exc:
            last_exc = exc
            raw_preview = repr(raw[:200]) if raw else "<no response>"
            print(
                f"\n[ERROR] customer_id={customer_id} attempt={attempt + 1} "
                f"error={type(exc).__name__}: {exc} | raw={raw_preview}",
                flush=True,
            )
            if attempt < 2:
                time.sleep(4)

    raise ValueError(f"All 3 attempts failed: {last_exc}")


# ── Persistence ────────────────────────────────────────────────────────────────

def save_conversations(conversations: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for conv in conversations:
            fh.write(json.dumps(conv, ensure_ascii=False) + "\n")


def build_record(profile: dict, turns: list[dict]) -> dict:
    scenario = profile.get("complaint_scenario", "happy_path")
    return {
        "conversation_id":    str(uuid.uuid4()),
        "customer_id":        profile.get("customer_id", str(uuid.uuid4())),
        "name":               profile.get("customer_name", ""),
        "language":           profile.get("language", "english"),
        "scenario":           scenario,
        "emotion":            profile.get("expected_emotion", "neutral"),
        "customer_segment":   profile.get("customer_segment", "regular"),
        "image_path":         profile.get("image_path"),
        "turns":              turns,
        "expected_intent":    SCENARIO_TO_INTENT.get(scenario, "general_inquiry"),
        "expected_resolution": profile.get("expected_resolution", "explanation"),
        "generated_at":       datetime.now(timezone.utc).isoformat(),
    }


# ── Stats ──────────────────────────────────────────────────────────────────────

def print_stats(conversations: list[dict]) -> None:
    from collections import Counter
    sc   = Counter(c["scenario"]          for c in conversations)
    lang = Counter(c["language"]          for c in conversations)
    seg  = Counter(c["customer_segment"]  for c in conversations)
    lens = [len(c["turns"]) for c in conversations]
    avg  = sum(lens) / len(lens) if lens else 0

    print("\n── Scenario distribution ─────────────────────")
    for k, v in sorted(sc.items()):
        print(f"  {k:<22} {v:>4}")
    print("\n── Language distribution ─────────────────────")
    for k, v in sorted(lang.items()):
        print(f"  {k:<12} {v:>4}")
    print("\n── Customer segments ─────────────────────────")
    for k, v in sorted(seg.items()):
        print(f"  {k:<12} {v:>4}")
    print(f"\n── Avg turns per conversation: {avg:.1f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Resolve GROQ_API_KEY from env or .env file
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
            "Set it as an environment variable or add it to .env at the project root."
        )
        sys.exit(1)

    try:
        from groq import Groq
    except ImportError:
        print("ERROR: groq package not installed. Run: pip install groq")
        sys.exit(1)

    client = Groq(api_key=groq_api_key)

    # Load and proportionally sample profiles
    profiles = load_profiles()
    sampled  = sample_proportionally(profiles, TARGET_COUNT)
    print(f"Sampled {len(sampled)} profiles (proportional by scenario)\n")

    # Resume: load any conversations already saved to disk
    conversations: list[dict] = []
    done_ids: set[str] = set()

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    conv = json.loads(line)
                    conversations.append(conv)
                    done_ids.add(conv["customer_id"])
                except json.JSONDecodeError:
                    pass
        if conversations:
            print(f"Resuming: {len(conversations)} conversations already exist\n")

    # Filter out already-processed customer IDs
    pending = [p for p in sampled if p.get("customer_id") not in done_ids]
    if not pending:
        print("All conversations already generated. Nothing to do.")
        print_stats(conversations)
        return

    print(f"Generating {len(pending)} conversations in batches of {BATCH_SIZE}…")
    print(f"Progress saved every {SAVE_INTERVAL} conversations.\n")

    errors        = 0
    last_save_at  = len(conversations)
    total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch       = pending[batch_start : batch_start + BATCH_SIZE]

        print(
            f"Batch {batch_idx + 1:>3}/{total_batches}  "
            f"[{batch_start + 1}–{batch_start + len(batch):>3}]  ",
            end="", flush=True,
        )

        for profile in batch:
            try:
                turns  = generate_conversation(client, profile)
                record = build_record(profile, turns)
                conversations.append(record)
                print(".", end="", flush=True)
            except Exception as exc:
                errors += 1
                print(f"E", end="", flush=True)
                # Log the error without stopping the run
                _ = exc  # suppress linter warning

        print()  # newline after batch progress dots

        # Save progress every SAVE_INTERVAL conversations
        if len(conversations) - last_save_at >= SAVE_INTERVAL or batch_idx == total_batches - 1:
            save_conversations(conversations, OUTPUT_FILE)
            last_save_at = len(conversations)
            print(f"  ↳ Saved {len(conversations)} conversations to {OUTPUT_FILE.name}")

        # Rate-limit courtesy sleep (skip after final batch)
        if batch_idx < total_batches - 1:
            time.sleep(BATCH_SLEEP)

    # Final flush (idempotent if already saved above)
    save_conversations(conversations, OUTPUT_FILE)

    print(f"\n{'─' * 52}")
    print(f"Done — {len(conversations)} conversations written to:")
    print(f"  {OUTPUT_FILE.relative_to(REPO_ROOT)}")
    if errors:
        print(f"  Errors/skipped: {errors} profiles")
    print_stats(conversations)


if __name__ == "__main__":
    main()
