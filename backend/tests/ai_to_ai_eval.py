"""
backend/tests/ai_to_ai_eval.py

AI-to-AI evaluation harness for Druta Kart.

Flow per conversation:
  1. Minor LLM (llama-3.1-8b-instant) plays the customer.
  2. Real FastAPI at http://localhost:8000 plays the bot.
  3. Up to 3 turns of back-and-forth.
  4. Judge LLM (llama-3.3-70b-versatile) scores the conversation.
  5. Result saved to backend/data/eval_results.jsonl.

Usage:
  python tests/ai_to_ai_eval.py              # 100 conversations
  python tests/ai_to_ai_eval.py --count 20   # 20 conversations
  python tests/ai_to_ai_eval.py --count 0    # all conversations
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time

# Windows cp1252 fix — must be before any print() with Unicode
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent          # backend/tests/
BACKEND_DIR  = SCRIPT_DIR.parent                        # backend/
DATA_DIR     = BACKEND_DIR / "data"

CONVERSATIONS_PATH = DATA_DIR / "synthetic" / "conversations.jsonl"
EVAL_RESULTS_PATH  = DATA_DIR / "eval_results.jsonl"
EVAL_REPORT_PATH      = DATA_DIR / "eval_report.md"
EVAL_REPORT_HTML_PATH = DATA_DIR / "eval_report.html"

# ---------------------------------------------------------------------------
# Image pools — real images for eval (relative to repo root)
# ---------------------------------------------------------------------------

_REPO_ROOT = BACKEND_DIR.parent  # E:/Desktop/Projects/Druta-Kart

# Rotten produce = visually damaged → Vision LLM returns real_damage
_DAMAGED_DIRS = [
    _REPO_ROOT / "data/images/damaged_food/dataset/Test/rottenapples",
    _REPO_ROOT / "data/images/damaged_food/dataset/Test/rottenoranges",
    _REPO_ROOT / "data/images/damaged_food/dataset/Test/rottenbanana",
    _REPO_ROOT / "data/images/damaged_food/dataset/Test/rottentamto",
    _REPO_ROOT / "data/images/damaged_food/dataset/Test/rottenpatato",
]

# Fresh produce = no visible damage → Vision LLM returns misidentification
_MISIDENTIFICATION_DIRS = [
    _REPO_ROOT / "data/images/misidentification/dataset/freshapples",
    _REPO_ROOT / "data/images/misidentification/dataset/freshoranges",
    _REPO_ROOT / "data/images/misidentification/dataset/freshbanana",
    _REPO_ROOT / "data/images/misidentification/dataset/freshoranges",
]

# AI-generated fakes → Vision LLM returns ai_generated
_AI_GENERATED_DIRS = [
    _REPO_ROOT / "data/images/ai_generated_fake/test/FAKE",
]

# Pre-built flat lists loaded once at startup (populated in main())
_IMAGE_POOL: dict[str, list[Path]] = {
    "real_damage":      [],
    "misidentification": [],
    "ai_generated":     [],
}


# ---------------------------------------------------------------------------
# Mandatory fake_image_fraud test conversations (always run regardless of --count)
# ---------------------------------------------------------------------------

_MANDATORY_FRAUD_CONVS = [
    {
        "conversation_id": "fraud-test-001",
        "customer_id":     "fraud-test-user-001",
        "name":            "Rahul Sharma",
        "language":        "english",
        "scenario":        "fake_image_fraud",
        "emotion":         "angry",
        "customer_segment": "regular",
        "image_path":      None,
        "expected_intent":      "complaint_damaged_item",
        "expected_resolution":  "human_review",
    },
    {
        "conversation_id": "fraud-test-002",
        "customer_id":     "fraud-test-user-002",
        "name":            "Priya Nair",
        "language":        "hinglish",
        "scenario":        "fake_image_fraud",
        "emotion":         "upset",
        "customer_segment": "bulk",
        "image_path":      None,
        "expected_intent":      "complaint_damaged_item",
        "expected_resolution":  "human_review",
    },
    {
        "conversation_id": "fraud-test-003",
        "customer_id":     "fraud-test-user-003",
        "name":            "Ayesha Khan",
        "language":        "english",
        "scenario":        "fake_image_fraud",
        "emotion":         "neutral",
        "customer_segment": "vip",
        "image_path":      None,
        "expected_intent":      "complaint_damaged_item",
        "expected_resolution":  "human_review",
    },
]


def _build_image_pools() -> None:
    """Scan image folders and build flat lists of available image paths."""
    for pool_key, dirs in [
        ("real_damage",      _DAMAGED_DIRS),
        ("misidentification", _MISIDENTIFICATION_DIRS),
        ("ai_generated",     _AI_GENERATED_DIRS),
    ]:
        files: list[Path] = []
        for d in dirs:
            if d.exists():
                files.extend(p for p in d.iterdir()
                             if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        _IMAGE_POOL[pool_key] = files
        print(f"  image pool [{pool_key}]: {len(files)} images", flush=True)


def _pick_image_for_conv(conv: dict) -> Path | None:
    """Return a real image path for a damaged_product conversation.

    Priority:
      1. Use conv's own image_path if it resolves to a real file on disk.
      2. Infer the correct pool from the image_path string (rotten/fresh/FAKE).
      3. Fall back to real_damage pool (most common scenario).
    """
    raw_path: str | None = conv.get("image_path")

    # --- Priority 1: use the conversation's own image if it exists ---
    if raw_path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = _REPO_ROOT / raw_path
        if candidate.exists() and candidate.stat().st_size > 1024:
            return candidate

    # --- Priority 2: always fall back to real_damage (rotten food) ---
    # Damaged product evals test the complaint resolution path.
    # AI-generated / misidentification paths require dedicated test scenarios.
    pool = _IMAGE_POOL.get("real_damage", [])
    return random.choice(pool) if pool else None

# ---------------------------------------------------------------------------
# Models + pricing (per 1M tokens)
# ---------------------------------------------------------------------------

SIMULATOR_MODEL = "llama-3.1-8b-instant"
JUDGE_MODEL     = "llama-3.3-70b-versatile"

PRICE_PER_M = {
    SIMULATOR_MODEL: 0.06,
    JUDGE_MODEL:     0.59,
}

# ---------------------------------------------------------------------------
# Bot endpoint
# ---------------------------------------------------------------------------

BOT_BASE_URL  = "http://localhost:8000"
CHAT_ENDPOINT = f"{BOT_BASE_URL}/chat"
IMG_ENDPOINT  = f"{BOT_BASE_URL}/upload-image"

MAX_TURNS            = 3
INTER_CONV_SLEEP_SEC = 5

# ---------------------------------------------------------------------------
# Globals — updated during the run
# ---------------------------------------------------------------------------

_total_tokens: dict[str, int] = {SIMULATOR_MODEL: 0, JUDGE_MODEL: 0}
_total_api_calls   = 0
_rate_limit_hits   = 0

# Loaded from orders_map.json if available: customer_id → list of order dicts
_ORDERS_MAP: dict[str, list[dict]] = {}

ORDERS_MAP_PATH = DATA_DIR / "synthetic" / "orders_map.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_evaluated_ids() -> set[str]:
    """Return conversation_ids already present in eval_results.jsonl."""
    if not EVAL_RESULTS_PATH.exists():
        return set()
    ids: set[str] = set()
    with EVAL_RESULTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["conversation_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def _append_result(result: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with EVAL_RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def _groq_call(client: Groq, model: str, messages: list[dict], *, max_tokens: int = 512) -> str:
    """Call Groq and update global token + call counters. Retries once on rate limit."""
    global _total_api_calls, _rate_limit_hits

    for attempt in range(2):
        try:
            _total_api_calls += 1
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            usage = resp.usage
            if usage:
                _total_tokens[model] = _total_tokens.get(model, 0) + (usage.total_tokens or 0)
            return resp.choices[0].message.content.strip()

        except Exception as exc:
            err_str = str(exc).lower()
            if "rate" in err_str and attempt == 0:
                _rate_limit_hits += 1
                print(f"  [rate limit] sleeping 30s …", flush=True)
                time.sleep(30)
                continue
            raise


# ---------------------------------------------------------------------------
# Customer simulator
# ---------------------------------------------------------------------------

_SCENARIO_PROMPTS = {
    "damaged_product":  "Your recently delivered product was damaged. Complain about it.",
    "late_delivery":    "Your order is very late and hasn't arrived. Ask about it.",
    "wrong_item":       "You received the wrong item in your order. Complain.",
    "missing_item":     "Part of your order is missing. Ask for help.",
    "payment_issue":    "You were charged incorrectly or your payment failed. Ask for help.",
    "happy_path":       "You are a happy customer. Ask a general question about your order or products.",
    "fake_image_fraud": "Your product was damaged. You are uploading a photo as proof. Be insistent about getting a refund.",
}

_LANG_INSTRUCTION = {
    "english":   "Respond only in English.",
    "hindi":     "Respond only in Hindi (Devanagari script).",
    "hinglish":  "Respond in Hinglish — a natural mix of Hindi and English.",
    "manglish":  "Respond in Manglish — a natural mix of Malayalam and English.",
    "tamil":     "Respond in Tamil.",
    "telugu":    "Respond in Telugu.",
    "kannada":   "Respond in Kannada.",
    "bengali":   "Respond in Bengali.",
}


def _simulate_customer_turn(
    client: Groq,
    name: str,
    language: str,
    scenario: str,
    emotion: str,
    conversation_so_far: list[dict],
    turn_number: int,
    order_id: str | None = None,
) -> str:
    """Generate the next customer message using the simulator LLM."""
    scenario_desc = _SCENARIO_PROMPTS.get(scenario, "You have a general question.")
    lang_instr    = _LANG_INSTRUCTION.get(language, "Respond naturally.")

    order_hint = (
        f"Your order ID is {order_id}. Use this exact ID when the agent asks for it. "
        if order_id else ""
    )

    system = (
        f"You are {name}, a customer of Druta Kart (an Indian quick-commerce app). "
        f"Your current emotion is {emotion}. "
        f"{scenario_desc} "
        f"{order_hint}"
        f"{lang_instr} "
        "Keep your message short (1-3 sentences). "
        "Do not resolve the issue yourself — wait for the support agent. "
        "If this is not the first turn, continue naturally from the conversation so far."
    )

    messages: list[dict] = [{"role": "system", "content": system}]
    for turn in conversation_so_far:
        role = "user" if turn["role"] == "customer" else "assistant"
        messages.append({"role": role, "content": turn["content"]})

    if turn_number == 1:
        messages.append({"role": "user", "content": "Start the conversation with your complaint or question."})
    else:
        messages.append({"role": "user", "content": "Continue the conversation as the customer. Respond to the agent's last message."})

    return _groq_call(client, SIMULATOR_MODEL, messages, max_tokens=200)


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are a strict QA evaluator for Druta Kart, an Indian quick-commerce customer support AI.
You will be given a full customer-support conversation and must score it on 6 dimensions.

Scoring rules:
- intent_correct: Score 1 if bot intent matches expected intent OR is a valid equivalent.
  IMPORTANT: Always follow these mapping rules EXACTLY. Do NOT override them based on what the customer said in the transcript.
  Mapping rules (bot_intent → acceptable expected_intent values):
    'complaint'      → complaint_damaged_item, complaint_missing_item, complaint_wrong_item, complaint_expired, complaint_late_delivery, complaint
    'payment'        → inquiry_payment, payment_issue, payment
    'general'        → general_inquiry, happy_path, general
    'order_tracking' → order_tracking, order_status, general_inquiry, happy_path, general
    'delivery'       → complaint_late_delivery, late_delivery, not_delivered, delivery
    'late_delivery'  → complaint_late_delivery, late_delivery, delivery
    'delivery'  is NOT correct for: complaint_missing_item, complaint_wrong_item, complaint_damaged_item
  STRICT RULE: Score 0 ONLY if it is a completely wrong category — e.g. payment routed to delivery, or complaint routed to payment.
  EXAMPLE: bot_intent='order_tracking' with scenario='happy_path' → intent_correct=1 (because order_tracking maps to happy_path)
  EXAMPLE: bot_intent='complaint' with scenario='happy_path' → intent_correct=1 (complaint is plausible for a customer with an issue)
  EXAMPLE: bot_intent='general' with scenario='late_delivery' → intent_correct=0 (completely wrong — bot ignored the delivery issue)
  EXAMPLE: bot_intent='complaint' with scenario='fake_image_fraud' → intent_correct=1 (customer is complaining, bot correctly routes it)
- resolution_achieved : 1 if the customer's issue was clearly resolved or a concrete next step promised, else 0. For happy_path and general scenarios: score resolution_achieved=1 if bot gave a warm helpful answer even without calling a tool. A friendly informative response counts as resolved for general inquiries. For late_delivery and delivery scenarios: score resolution_achieved=1 if bot acknowledged the delay, escalated to the dispatch team, or offered compensation — a clear escalation promise counts as resolved. For fake_image_fraud scenarios: score resolution_achieved=1 if bot says the case will be reviewed by a quality assurance or support team within a time frame (e.g. 2 hours) — this is the correct handling for suspicious images.
- tone_appropriate    : 1-5 (5 = very empathetic and polite, 1 = rude or dismissive)
- language_correct    : 1 if the bot responded in the SAME language as the customer (or English if customer used English/Hinglish/Kanglish/Manglish), else 0
- hallucination_detected : 1 if the bot stated specific false facts (fake order IDs, invented policies), else 0
- offer_within_caps   : 1 if any offer/compensation mentioned is within ₹200 wallet credit OR 35% discount OR 2 free items. If no offer was made, score 1.

Return ONLY valid JSON with exactly these keys (no markdown, no explanation):
{
  "intent_correct": 0,
  "resolution_achieved": 0,
  "tone_appropriate": 3,
  "language_correct": 1,
  "hallucination_detected": 0,
  "offer_within_caps": 1,
  "reasoning": "brief explanation"
}
"""


def _judge_conversation(
    client: Groq,
    conversation: list[dict],
    scenario: str,
    language: str,
    expected_intent: str,
    bot_intent: str | None,
) -> dict[str, Any]:
    """Ask the judge LLM to score a completed conversation."""
    transcript_lines = []
    for turn in conversation:
        role = "Customer" if turn["role"] == "customer" else "Bot"
        transcript_lines.append(f"{role}: {turn['content']}")
    transcript = "\n".join(transcript_lines)

    user_msg = (
        f"Scenario: {scenario}\n"
        f"Customer language: {language}\n"
        f"Expected intent category: {expected_intent}\n"
        f"Bot reported intent: {bot_intent or 'unknown'}\n\n"
        f"Conversation transcript:\n{transcript}"
    )

    raw = _groq_call(
        client,
        JUDGE_MODEL,
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=300,
    )

    # Strip optional markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback neutral scores
        scores = {
            "intent_correct": 0,
            "resolution_achieved": 0,
            "tone_appropriate": 3,
            "language_correct": 1,
            "hallucination_detected": 0,
            "offer_within_caps": 1,
            "reasoning": f"parse_error: {raw[:200]}",
        }

    return scores


# ---------------------------------------------------------------------------
# Single conversation evaluation
# ---------------------------------------------------------------------------

async def _evaluate_conversation(
    groq_client: Groq,
    http_client: httpx.AsyncClient,
    conv: dict[str, Any],
    conv_index: int,
    total: int,
) -> dict[str, Any]:
    conv_id    = conv["conversation_id"]
    name       = conv.get("name", "Customer")
    language   = conv.get("language", "english")
    scenario   = conv.get("scenario", "happy_path")
    emotion    = conv.get("emotion", "neutral")
    image_path = conv.get("image_path")
    user_id    = conv.get("customer_id", conv_id)
    session_id = f"eval_{conv_id[:8]}"

    expected_intent     = conv.get("expected_intent", "general_inquiry")
    expected_resolution = conv.get("expected_resolution", "explanation")

    # ── Pick a real order for this customer from the seeded orders map ───────
    customer_orders = _ORDERS_MAP.get(user_id, [])
    selected_order  = random.choice(customer_orders) if customer_orders else None
    real_order_id   = selected_order["order_id"] if selected_order else None

    print(
        f"\n[{conv_index}/{total}] {name} | {scenario} | {language}"
        + (f" | order={real_order_id}" if real_order_id else " | no order seeded"),
        flush=True,
    )

    dialogue: list[dict] = []   # {"role": "customer"|"bot", "content": str}
    bot_intent: str | None = None
    bot_resolved           = False
    image_uploaded         = False
    conv_tokens_simulator  = 0
    conv_tokens_judge      = 0
    locked_image_validation: str | None = None
    conv_start_time    = time.monotonic()
    fraud_flagged_conv = False
    human_handoff_conv = False
    customer_segment   = conv.get("customer_segment", "unknown")

    # ── Upload image if damaged_product or fake_image_fraud ──────────────────
    uploaded_image_path: str | None = None
    if scenario in ("damaged_product", "fake_image_fraud"):
        if scenario == "fake_image_fraud":
            # Deliberately upload an AI-generated fake image
            pool = _IMAGE_POOL.get("ai_generated", [])
            real_img = random.choice(pool) if pool else None
        else:
            real_img = _pick_image_for_conv(conv)
        if real_img:
            try:
                ext = real_img.suffix.lower().lstrip(".")
                mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
                with real_img.open("rb") as img_f:
                    resp = await http_client.post(
                        IMG_ENDPOINT,
                        params={"session_id": session_id},
                        files={"file": (real_img.name, img_f, mime)},
                        timeout=30,
                    )
                if resp.status_code == 200:
                    uploaded_image_path = resp.json().get("image_path")
                    image_uploaded = True
                    print(f"  image → {real_img.name} ({mime}) → uploaded as {uploaded_image_path}", flush=True)
                else:
                    print(f"  image upload failed ({resp.status_code}): {resp.text[:100]}", flush=True)
            except Exception as exc:
                print(f"  image upload error: {exc}", flush=True)
        else:
            print(f"  no image available for this conversation, skipping upload", flush=True)

    # ── Conversation loop (up to MAX_TURNS) ──────────────────────────────────
    tokens_before_sim = _total_tokens.get(SIMULATOR_MODEL, 0)

    for turn_num in range(1, MAX_TURNS + 1):
        # 1. Simulator generates customer message
        customer_msg = _simulate_customer_turn(
            groq_client, name, language, scenario, emotion,
            dialogue, turn_num, order_id=real_order_id,
        )
        dialogue.append({"role": "customer", "content": customer_msg})
        print(f"  T{turn_num} customer: {customer_msg[:80]}…" if len(customer_msg) > 80
              else f"  T{turn_num} customer: {customer_msg}", flush=True)

        # 2. Call bot /chat
        payload: dict[str, Any] = {
            "user_id":    user_id,
            "session_id": session_id,
            "message":    customer_msg,
        }
        if real_order_id:
            payload["order_id"] = real_order_id
        if uploaded_image_path:
            payload["image_path"] = uploaded_image_path
        if turn_num > 1 and locked_image_validation is not None:
            payload["image_validation_result"] = locked_image_validation

        try:
            resp = await http_client.post(CHAT_ENDPOINT, json=payload, timeout=120)
            if resp.status_code == 200:
                data       = resp.json()
                bot_reply  = data.get("response", "")
                bot_intent = data.get("intent", bot_intent)
                bot_resolved = data.get("resolved", False)
                fraud_flagged_conv = data.get("fraud_flagged", False) or fraud_flagged_conv
                human_handoff_conv = data.get("human_handoff", False) or human_handoff_conv
                if turn_num == 1 and data.get("image_validation_result"):
                    locked_image_validation = data["image_validation_result"]
            else:
                bot_reply = f"[HTTP {resp.status_code}]"
        except Exception as exc:
            bot_reply = f"[connection error: {exc}]"

        dialogue.append({"role": "bot", "content": bot_reply})
        print(f"  T{turn_num} bot:      {bot_reply[:80]}…" if len(bot_reply) > 80
              else f"  T{turn_num} bot:      {bot_reply}", flush=True)

        # Stop early if bot says resolved
        if bot_resolved and turn_num >= 2:
            break

    conv_tokens_simulator = _total_tokens.get(SIMULATOR_MODEL, 0) - tokens_before_sim

    # ── Judge scores ─────────────────────────────────────────────────────────
    tokens_before_judge = _total_tokens.get(JUDGE_MODEL, 0)
    scores = _judge_conversation(
        groq_client, dialogue, scenario, language,
        expected_intent, bot_intent,
    )
    conv_tokens_judge = _total_tokens.get(JUDGE_MODEL, 0) - tokens_before_judge

    print(
        f"  scores → intent={scores.get('intent_correct')} "
        f"resolved={scores.get('resolution_achieved')} "
        f"tone={scores.get('tone_appropriate')} "
        f"lang={scores.get('language_correct')} "
        f"halluc={scores.get('hallucination_detected')} "
        f"offer_ok={scores.get('offer_within_caps')}",
        flush=True,
    )

    result: dict[str, Any] = {
        "conversation_id":     conv_id,
        "customer_id":         user_id,
        "name":                name,
        "language":            language,
        "scenario":            scenario,
        "emotion":             emotion,
        "expected_intent":     expected_intent,
        "expected_resolution": expected_resolution,
        "bot_intent":          bot_intent,
        "bot_resolved":        bot_resolved,
        "image_uploaded":      image_uploaded,
        "turns_taken":         len([d for d in dialogue if d["role"] == "customer"]),
        "dialogue":            dialogue,
        "scores":              scores,
        "tokens_simulator":    conv_tokens_simulator,
        "tokens_judge":        conv_tokens_judge,
        "evaluated_at":        datetime.now(timezone.utc).isoformat(),
        "latency_ms":          round((time.monotonic() - conv_start_time) * 1000),
        "fraud_flagged":       fraud_flagged_conv,
        "human_handoff":       human_handoff_conv,
        "customer_segment":    customer_segment,
    }

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _build_report(results: list[dict[str, Any]], elapsed_sec: float) -> tuple[str, str, str]:
    """Build terminal summary, Markdown report, and HTML report."""
    n = len(results)
    if n == 0:
        empty = "No results to report."
        return empty, "# No results to report.", "<p>No results.</p>"

    def _avg(key: str) -> float:
        vals = [r["scores"].get(key, 0) for r in results if "scores" in r]
        return sum(vals) / len(vals) if vals else 0.0

    resolution_rate  = _avg("resolution_achieved") * 100
    intent_acc       = _avg("intent_correct") * 100
    lang_acc         = _avg("language_correct") * 100
    avg_tone         = _avg("tone_appropriate")
    halluc_rate      = _avg("hallucination_detected") * 100
    offer_compliance = _avg("offer_within_caps") * 100

    fraud_total    = sum(1 for r in results if r.get("fraud_flagged"))
    handoff_total  = sum(1 for r in results if r.get("human_handoff"))
    latencies      = [r["latency_ms"] for r in results if r.get("latency_ms")]
    avg_latency_ms = round(sum(latencies) / len(latencies)) if latencies else 0

    tok_sim   = _total_tokens.get(SIMULATOR_MODEL, 0)
    tok_judge = _total_tokens.get(JUDGE_MODEL, 0)
    cost_sim   = tok_sim   / 1_000_000 * PRICE_PER_M[SIMULATOR_MODEL]
    cost_judge = tok_judge / 1_000_000 * PRICE_PER_M[JUDGE_MODEL]
    total_cost = cost_sim + cost_judge
    total_tok  = tok_sim + tok_judge

    # ── Scenario breakdown ─────────────────────────────────────────────────
    scenario_stats: dict[str, dict] = {}
    for r in results:
        sc = r.get("scenario", "unknown")
        if sc not in scenario_stats:
            scenario_stats[sc] = {"n": 0, "resolved": 0, "intent": 0,
                                   "fraud": 0, "handoff": 0, "latency": []}
        scenario_stats[sc]["n"]        += 1
        scenario_stats[sc]["resolved"] += r["scores"].get("resolution_achieved", 0)
        scenario_stats[sc]["intent"]   += r["scores"].get("intent_correct", 0)
        scenario_stats[sc]["fraud"]    += 1 if r.get("fraud_flagged") else 0
        scenario_stats[sc]["handoff"]  += 1 if r.get("human_handoff") else 0
        if r.get("latency_ms"):
            scenario_stats[sc]["latency"].append(r["latency_ms"])

    # ── Segment breakdown ──────────────────────────────────────────────────
    segment_stats: dict[str, dict] = {}
    for r in results:
        seg = r.get("customer_segment", "unknown")
        if seg not in segment_stats:
            segment_stats[seg] = {"n": 0, "resolved": 0}
        segment_stats[seg]["n"]        += 1
        segment_stats[seg]["resolved"] += r["scores"].get("resolution_achieved", 0)

    # ── Terminal output ────────────────────────────────────────────────────
    terminal = (
        "\nDRUTA KART AI EVALUATION REPORT\n"
        "================================\n"
        f"Total conversations tested: {n}\n"
        f"Overall resolution rate:    {resolution_rate:.1f}%\n"
        f"Intent accuracy:            {intent_acc:.1f}%\n"
        f"Language accuracy:          {lang_acc:.1f}%\n"
        f"Avg tone score:             {avg_tone:.1f}/5\n"
        f"Hallucination rate:         {halluc_rate:.1f}%\n"
        f"Offer compliance:           {offer_compliance:.1f}%\n"
        f"Fraud detected:             {fraud_total}\n"
        f"Human handoffs:             {handoff_total}\n"
        f"Avg latency per chat:       {avg_latency_ms:,} ms\n"
        f"Total tokens used:          {total_tok:,}\n"
        f"Estimated cost:             ${total_cost:.4f}\n"
        f"Total API calls:            {_total_api_calls}\n"
        f"Rate limit hits:            {_rate_limit_hits}\n"
        f"Elapsed time:               {elapsed_sec/60:.1f} min\n"
    )

    # ── Markdown report ────────────────────────────────────────────────────
    sc_rows = []
    for sc, st in sorted(scenario_stats.items()):
        avg_lat = round(sum(st["latency"]) / len(st["latency"])) if st["latency"] else 0
        sc_rows.append(
            f"| {sc:<20} | {st['n']:>5} | "
            f"{st['resolved']/st['n']*100:>9.1f}% | "
            f"{st['intent']/st['n']*100:>9.1f}% | "
            f"{st['fraud']:>13} | "
            f"{st['handoff']:>14} | "
            f"{avg_lat:>11,} ms |"
        )

    seg_rows = []
    for seg, st in sorted(segment_stats.items()):
        seg_rows.append(
            f"| {seg:<18} | {st['n']:>5} | "
            f"{st['resolved']/st['n']*100:>9.1f}% |"
        )

    gen_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    md = f"""# Druta Kart AI Evaluation Report

Generated: {gen_time}

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| Conversations tested | {n} |
| Resolution rate | **{resolution_rate:.1f}%** |
| Intent accuracy | **{intent_acc:.1f}%** |
| Language accuracy | {lang_acc:.1f}% |
| Avg tone score | {avg_tone:.1f} / 5 |
| Hallucination rate | {halluc_rate:.1f}% |
| Offer compliance | {offer_compliance:.1f}% |
| Fraud detected & escalated | {fraud_total} |
| Human handoffs | {handoff_total} |
| Avg latency per chat | {avg_latency_ms:,} ms |

---

## 2. Token Usage & Cost

| Model | Tokens | Cost |
|-------|--------|------|
| {SIMULATOR_MODEL} | {tok_sim:,} | ${cost_sim:.4f} |
| {JUDGE_MODEL} | {tok_judge:,} | ${cost_judge:.4f} |
| **Total** | **{total_tok:,}** | **${total_cost:.4f}** |

Total API calls: {_total_api_calls}
Rate limit hits: {_rate_limit_hits}
Elapsed: {elapsed_sec/60:.1f} min

---

## 3. Scenario Breakdown

| Scenario | Count | Resolution | Intent Acc | Fraud Detected | Human Handoff | Avg Latency |
|----------|-------|-----------|-----------|---------------|--------------|-------------|
{chr(10).join(sc_rows)}

---

## 4. Customer Segment Breakdown

| Segment | Count | Resolution |
|---------|-------|-----------|
{chr(10).join(seg_rows)}

---

## 5. Sample Conversations

"""
    # Add up to 1 sample dialogue per scenario
    shown_scenarios: set[str] = set()
    for r in results:
        sc = r.get("scenario", "")
        if sc in shown_scenarios:
            continue
        shown_scenarios.add(sc)
        md += f"### {sc.replace('_', ' ').title()} — {r.get('name', '')} ({r.get('language', '')})\n\n"
        md += f"*Scores: resolution={r['scores'].get('resolution_achieved')} intent={r['scores'].get('intent_correct')} tone={r['scores'].get('tone_appropriate')}/5*\n\n"
        for turn in r.get("dialogue", []):
            role = "**Customer:**" if turn["role"] == "customer" else "**Bot:**"
            md += f"{role} {turn['content'][:300]}\n\n"
        md += "---\n\n"

    # ── HTML report ────────────────────────────────────────────────────────
    def _pct_color(pct: float) -> str:
        if pct >= 85: return "#27ae60"
        if pct >= 70: return "#f39c12"
        return "#e74c3c"

    sc_html_rows = ""
    for sc, st in sorted(scenario_stats.items()):
        res_pct = st['resolved']/st['n']*100
        avg_lat = round(sum(st["latency"])/len(st["latency"])) if st["latency"] else 0
        sc_html_rows += f"""
        <tr>
          <td>{sc.replace('_',' ').title()}</td>
          <td style="text-align:center">{st['n']}</td>
          <td style="text-align:center;color:{_pct_color(res_pct)};font-weight:bold">{res_pct:.1f}%</td>
          <td style="text-align:center">{st['intent']/st['n']*100:.1f}%</td>
          <td style="text-align:center;color:{'#e74c3c' if st['fraud'] else '#27ae60'};font-weight:bold">{st['fraud']}</td>
          <td style="text-align:center">{st['handoff']}</td>
          <td style="text-align:center">{avg_lat:,} ms</td>
        </tr>"""

    seg_html_rows = ""
    for seg, st in sorted(segment_stats.items()):
        res_pct = st['resolved']/st['n']*100
        seg_html_rows += f"""
        <tr>
          <td>{seg.replace('_',' ').title()}</td>
          <td style="text-align:center">{st['n']}</td>
          <td style="text-align:center;color:{_pct_color(res_pct)};font-weight:bold">{res_pct:.1f}%</td>
        </tr>"""

    # Sample dialogues HTML
    dialogue_html = ""
    shown_sc2: set[str] = set()
    for r in results:
        sc = r.get("scenario", "")
        if sc in shown_sc2:
            continue
        shown_sc2.add(sc)
        scores = r.get("scores", {})
        dialogue_html += f"""
        <div class="dialogue-block">
          <h3>{sc.replace('_',' ').title()} &mdash; {r.get('name','')} <span class="lang-badge">{r.get('language','')}</span></h3>
          <p class="scores">Resolution: <b>{scores.get('resolution_achieved','?')}</b> &nbsp;|&nbsp;
             Intent: <b>{scores.get('intent_correct','?')}</b> &nbsp;|&nbsp;
             Tone: <b>{scores.get('tone_appropriate','?')}/5</b> &nbsp;|&nbsp;
             Fraud: <b>{'Yes' if r.get('fraud_flagged') else 'No'}</b> &nbsp;|&nbsp;
             Handoff: <b>{'Yes' if r.get('human_handoff') else 'No'}</b></p>
          <div class="chat">"""
        for turn in r.get("dialogue", []):
            css_class = "customer-msg" if turn["role"] == "customer" else "bot-msg"
            label = "Customer" if turn["role"] == "customer" else "Druta Kart Bot"
            content = turn["content"][:400].replace("<", "&lt;").replace(">", "&gt;")
            dialogue_html += f"""
            <div class="{css_class}">
              <span class="speaker">{label}</span>
              <p>{content}</p>
            </div>"""
        dialogue_html += """
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Druta Kart AI Evaluation Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa; color: #2d3436; }}
  .page {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}
  h1 {{ font-size: 2rem; color: #6c5ce7; margin-bottom: 4px; }}
  .subtitle {{ color: #636e72; font-size: 0.95rem; margin-bottom: 32px; }}
  h2 {{ font-size: 1.3rem; color: #2d3436; border-left: 4px solid #6c5ce7;
        padding-left: 12px; margin: 32px 0 16px; }}
  h3 {{ font-size: 1.05rem; color: #2d3436; margin-bottom: 8px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .kpi {{ background: #fff; border-radius: 10px; padding: 20px 16px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.07); text-align: center; }}
  .kpi .val {{ font-size: 2rem; font-weight: 700; color: #6c5ce7; }}
  .kpi .lbl {{ font-size: 0.78rem; color: #636e72; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi.green .val {{ color: #27ae60; }}
  .kpi.red   .val {{ color: #e74c3c; }}
  .kpi.amber .val {{ color: #f39c12; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
           border-radius: 10px; overflow: hidden;
           box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 32px; }}
  th {{ background: #6c5ce7; color: #fff; padding: 12px 14px; text-align: left; font-size: 0.85rem; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #f0f0f0; font-size: 0.9rem; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f8f9ff; }}
  .dialogue-block {{ background: #fff; border-radius: 10px; padding: 20px 24px;
                     box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 24px; }}
  .lang-badge {{ background: #6c5ce7; color: #fff; font-size: 0.72rem;
                 padding: 2px 8px; border-radius: 10px; margin-left: 8px;
                 vertical-align: middle; }}
  .scores {{ color: #636e72; font-size: 0.85rem; margin: 6px 0 12px; }}
  .chat {{ display: flex; flex-direction: column; gap: 10px; }}
  .customer-msg, .bot-msg {{ max-width: 82%; padding: 10px 14px; border-radius: 12px; }}
  .customer-msg {{ background: #dfe6e9; align-self: flex-start; }}
  .bot-msg {{ background: #ede9fe; align-self: flex-end; }}
  .speaker {{ display: block; font-size: 0.72rem; font-weight: 600;
              color: #636e72; margin-bottom: 4px; text-transform: uppercase; }}
  .bot-msg .speaker {{ color: #6c5ce7; }}
  .customer-msg p, .bot-msg p {{ font-size: 0.9rem; line-height: 1.5; }}
  .footer {{ text-align: center; color: #b2bec3; font-size: 0.8rem; margin-top: 40px; }}
  @media print {{
    body {{ background: #fff; }}
    .page {{ max-width: 100%; padding: 16px; }}
    .kpi {{ box-shadow: none; border: 1px solid #ddd; }}
    table {{ box-shadow: none; }}
    .dialogue-block {{ box-shadow: none; border: 1px solid #ddd; page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="page">

  <h1>Druta Kart</h1>
  <p class="subtitle">AI Customer Support &mdash; Evaluation Report &mdash; Generated: {gen_time}</p>

  <h2>1. Executive Summary</h2>
  <div class="kpi-grid">
    <div class="kpi {'green' if resolution_rate>=85 else 'amber' if resolution_rate>=70 else 'red'}">
      <div class="val">{resolution_rate:.1f}%</div><div class="lbl">Resolution Rate</div></div>
    <div class="kpi {'green' if intent_acc>=85 else 'amber' if intent_acc>=70 else 'red'}">
      <div class="val">{intent_acc:.1f}%</div><div class="lbl">Intent Accuracy</div></div>
    <div class="kpi green">
      <div class="val">{lang_acc:.1f}%</div><div class="lbl">Language Accuracy</div></div>
    <div class="kpi {'green' if avg_tone>=4.5 else 'amber'}">
      <div class="val">{avg_tone:.1f}<span style="font-size:1rem">/5</span></div><div class="lbl">Avg Tone Score</div></div>
    <div class="kpi {'green' if halluc_rate==0 else 'red'}">
      <div class="val">{halluc_rate:.1f}%</div><div class="lbl">Hallucination Rate</div></div>
    <div class="kpi green">
      <div class="val">{offer_compliance:.1f}%</div><div class="lbl">Offer Compliance</div></div>
    <div class="kpi {'red' if fraud_total>0 else 'green'}">
      <div class="val">{fraud_total}</div><div class="lbl">Fraud Detected</div></div>
    <div class="kpi amber">
      <div class="val">{handoff_total}</div><div class="lbl">Human Handoffs</div></div>
    <div class="kpi">
      <div class="val">{n}</div><div class="lbl">Conversations</div></div>
    <div class="kpi">
      <div class="val">{avg_latency_ms:,}<span style="font-size:1rem">ms</span></div><div class="lbl">Avg Latency</div></div>
  </div>

  <h2>2. Token Usage &amp; Cost</h2>
  <table>
    <tr><th>Model</th><th>Tokens</th><th>Cost</th></tr>
    <tr><td>{SIMULATOR_MODEL}</td><td>{tok_sim:,}</td><td>${cost_sim:.4f}</td></tr>
    <tr><td>{JUDGE_MODEL}</td><td>{tok_judge:,}</td><td>${cost_judge:.4f}</td></tr>
    <tr><td><b>Total</b></td><td><b>{total_tok:,}</b></td><td><b>${total_cost:.4f}</b></td></tr>
  </table>
  <p style="margin:-24px 0 32px;color:#636e72;font-size:0.88rem">
    Total API calls: <b>{_total_api_calls}</b> &nbsp;|&nbsp;
    Rate limit hits: <b>{_rate_limit_hits}</b> &nbsp;|&nbsp;
    Elapsed: <b>{elapsed_sec/60:.1f} min</b>
  </p>

  <h2>3. Scenario Breakdown</h2>
  <table>
    <tr>
      <th>Scenario</th><th>Count</th><th>Resolution</th><th>Intent Acc</th>
      <th>Fraud Detected</th><th>Human Handoff</th><th>Avg Latency</th>
    </tr>
    {sc_html_rows}
  </table>

  <h2>4. Customer Segment Breakdown</h2>
  <table>
    <tr><th>Segment</th><th>Count</th><th>Resolution</th></tr>
    {seg_html_rows}
  </table>

  <h2>5. Sample Conversations</h2>
  {dialogue_html}

  <div class="footer">
    Druta Kart AI Evaluation &mdash; Powered by Groq (llama-3.3-70b-versatile + llama-4-scout) &mdash; {gen_time}
  </div>
</div>
</body>
</html>"""

    return terminal, md, html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(count: int, seed: int = 42) -> None:
    global _total_tokens, _total_api_calls, _rate_limit_hits, _ORDERS_MAP

    load_dotenv(BACKEND_DIR.parent / ".env")
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        sys.exit("GROQ_API_KEY not set in environment / .env")

    # Build image pools from real image folders
    print("Building image pools…", flush=True)
    _build_image_pools()

    # Load seeded orders map if present
    if ORDERS_MAP_PATH.exists():
        with ORDERS_MAP_PATH.open(encoding="utf-8") as f:
            _ORDERS_MAP = json.load(f)
        print(f"Loaded orders_map.json — {len(_ORDERS_MAP)} customers with real order IDs")
    else:
        print("orders_map.json not found — simulator will invent order IDs")

    groq_client = Groq(api_key=api_key)

    # Load conversations
    if not CONVERSATIONS_PATH.exists():
        sys.exit(f"conversations.jsonl not found at {CONVERSATIONS_PATH}")

    with CONVERSATIONS_PATH.open(encoding="utf-8") as f:
        all_convs = [json.loads(line) for line in f if line.strip()]

    random.seed(seed)
    if count and count > 0:
        all_convs = random.sample(all_convs, min(count, len(all_convs)))

    # Resume: skip already evaluated
    done_ids = _load_evaluated_ids()
    pending  = [c for c in all_convs if c["conversation_id"] not in done_ids]
    # Always include mandatory fraud test conversations (skip if already done)
    fraud_pending = [c for c in _MANDATORY_FRAUD_CONVS if c["conversation_id"] not in done_ids]
    pending = pending + fraud_pending
    total = len(all_convs) + len(fraud_pending)

    print(f"Conversations: {total} selected, {len(done_ids)} already done, {len(pending)} to run")

    if not pending:
        print("Nothing to do — all conversations already evaluated.")
        return

    results_this_run: list[dict] = []
    start_time = time.monotonic()

    async with httpx.AsyncClient() as http_client:
        for idx, conv in enumerate(pending, start=1):
            try:
                result = await _evaluate_conversation(
                    groq_client, http_client, conv,
                    conv_index=len(done_ids) + idx,
                    total=total,
                )
                _append_result(result)
                results_this_run.append(result)
            except Exception as exc:
                print(f"  ERROR evaluating {conv['conversation_id']}: {exc}", flush=True)

            if idx < len(pending):
                time.sleep(INTER_CONV_SLEEP_SEC)

    elapsed = time.monotonic() - start_time

    # Load ALL results for the full report (including previous runs)
    all_results: list[dict] = []
    if EVAL_RESULTS_PATH.exists():
        with EVAL_RESULTS_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    terminal_report, md_report, html_report = _build_report(all_results, elapsed)
    print(terminal_report)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT_PATH.write_text(md_report, encoding="utf-8")
    EVAL_REPORT_HTML_PATH.write_text(html_report, encoding="utf-8")
    print(f"Report saved → {EVAL_REPORT_PATH}")
    print(f"HTML report  → {EVAL_REPORT_HTML_PATH}")
    print(f"Results saved → {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-to-AI evaluation for Druta Kart")
    parser.add_argument(
        "--count", type=int, default=100,
        help="Number of conversations to evaluate (0 = all, default 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible conversation sampling (default 42)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.count, seed=args.seed))
