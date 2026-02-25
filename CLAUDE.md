# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Druta Kart** is a multilingual agentic AI customer support backend for quick commerce (India). It is a FastAPI service that uses LangGraph-orchestrated specialist agents, Groq-hosted LLMs, Sarvam for Indian-language STT/TTS, ChromaDB for RAG, and Supabase for persistence.

## Development Commands

### Setup
```bash
cd backend
cp ../.env.example ../.env   # then fill in API keys
pip install -r requirements.txt
```

### Run locally
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Run with Docker
```bash
docker compose up --build          # full stack (backend + frontend)
docker compose up backend          # backend only
```

### Tests
```bash
cd backend
pytest tests/                          # all tests
pytest tests/test_agents.py            # single test file
pytest tests/test_emotion.py -v        # verbose single file
pytest -k "test_hallucination"         # run tests matching a name
```

### Health check
```bash
curl http://localhost:8000/health
```

## Architecture

### Request Flow

```
HTTP/WebSocket → FastAPI (main.py)
  → NLP pipeline: language detection → emotion analysis → translation (if needed)
  → Supervisor (agents/supervisor.py) — LangGraph StateGraph
      → routes to specialist agent based on intent
  → Agent calls Tools (tools/) to execute actions
  → Hallucination Guard (agents/hallucination_guard.py) — sampled at 50% by default
  → RAG retrieval (rag/vector_store.py) — ChromaDB + sentence-transformers
  → Response: translate back → TTS (multimodal/tts_processor.py) if voice
  → log_interaction() (observability/logger.py) + Prometheus metrics
```

### Specialist Agents (`backend/agents/`)
| Agent | Responsibility |
|---|---|
| `supervisor.py` | LangGraph StateGraph — intent routing to specialist agents |
| `complaint_agent.py` | Handles complaints, triggers refunds/replacements |
| `order_agent.py` | Order status lookups |
| `dispatch_agent.py` | Delivery/dispatch issues and checklists |
| `fraud_escalation_agent.py` | Detects fraud patterns, escalates |
| `retention_agent.py` | At-risk customer retention, offer generation |
| `image_validation_agent.py` | Validates complaint images |
| `hallucination_guard.py` | Post-generation factuality check (sampled) |

### Tools (`backend/tools/`)
LangChain tools called by agents: `order_lookup_tool`, `refund_tool`, `replacement_tool`, `wallet_credit_tool`, `offer_generator_tool`, `dispatch_checklist_tool`.

### NLP Pipeline (`backend/nlp/`)
- `language_detector.py` — uses `lingua-language-detector` (supports Indian languages)
- `translator.py` — translates non-English input → English for LLM, back for response
- `emotion_analyzer.py` — HuggingFace transformers-based sentiment/emotion classification

### Multimodal (`backend/multimodal/`)
- `image_analyzer.py` — Groq Vision (llama-4-scout) for complaint image analysis
- `stt_processor.py` — Groq Whisper (`whisper-large-v3`) for speech-to-text
- `tts_processor.py` — Sarvam API for Indian-language text-to-speech

### RAG (`backend/rag/`)
- Knowledge base documents in `rag/knowledge_base/` (FAQ, refund policy, return policy)
- `ingest.py` — chunks and embeds documents into ChromaDB
- `embeddings.py` — sentence-transformers embeddings
- `vector_store.py` — ChromaDB client; chroma data persisted to `rag/chroma_db/` (Docker volume: `chroma_data`)

### Database (`backend/db/`)
- Supabase (Postgres) via `supabase-py`
- `models.py` — Pydantic data models
- `customer_repo.py` — customer profile CRUD
- `analytics_repo.py` — interaction analytics writes

### Retention (`backend/retention/`)
- `profile_scorer.py` — scores customer churn risk
- `offer_engine.py` — generates retention offers; **hardcoded safety caps from `config.py`** that LLM cannot override: max ₹200 wallet credit, max 35% discount, max 2 free items per complaint

### Prompts (`backend/prompts/`)
- `prompt_registry.py` — loads prompts by version key (`PROMPT_VERSION` env var, default `v1`)
- Prompt text files: `v1_system_prompt.txt`, `v1_hallucination_check.txt`
- To add a new prompt version, add a `v2_*.txt` file and update `prompt_registry.py`

### Observability (`backend/observability/`)
- `logger.py` — `structlog` JSON logger; use `log_interaction()` after every turn
- `metrics.py` — Prometheus counters/histograms (exposed on `/metrics`)

## Configuration

All config lives in `backend/config.py` as a `pydantic-settings` `Settings` object (imported as `settings`). Values are read from `.env`. Key fields:

- `GROQ_API_KEY`, `SARVAM_API_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `HF_TOKEN`
- `GROQ_TEXT_MODEL` (default: `llama-3.3-70b-versatile`)
- `GROQ_VISION_MODEL` (default: `meta-llama/llama-4-scout-17b-16e-instruct`)
- `GROQ_WHISPER_MODEL` (default: `whisper-large-v3`)
- `HALLUCINATION_CHECK_SAMPLING_RATE` (default: `0.5`)
- Offer safety caps: `MAX_WALLET_CREDIT_INR`, `MAX_DISCOUNT_PERCENT`, `MAX_FREE_ITEMS_PER_COMPLAINT`

## Key Constraints

- **Offer safety caps are hardcoded in `config.py`** and must not be LLM-controllable.
- Rate limiting is enforced by `slowapi`: 30 req/min per IP, 3 image uploads per session.
- CORS allows `localhost:3000` and `localhost:5173` only (add production origins explicitly).
- PyTorch is installed as CPU-only (`torch==2.5.1+cpu`) to keep the image lightweight.
