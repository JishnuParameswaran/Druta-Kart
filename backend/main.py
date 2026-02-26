"""
Druta Kart - Main FastAPI Application

Request flow:
  HTTP POST /chat or WebSocket /ws/{session_id}
    → NLP pipeline: language detection → emotion analysis → translate to English
    → Supervisor (LangGraph agents) → response
    → Translate response back to customer language
    → Optional TTS for voice responses
    → Structured logging + Prometheus metrics
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Druta Kart AI Support",
    description="Multilingual Agentic AI Customer Support for Quick Commerce",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Image upload config
# ---------------------------------------------------------------------------

_UPLOAD_DIR = Path(tempfile.gettempdir()) / "druta_kart_uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

# Per-session image upload counter (in-memory; resets on restart)
_session_image_counts: dict[str, int] = {}

# ---------------------------------------------------------------------------
# Order ID extraction
# ---------------------------------------------------------------------------

# Matches patterns like: ORD123456, ORD-ABC123, #ABC123XY
_ORDER_RE = re.compile(r"\b(ORD-?[A-Z0-9]{4,}|#[A-Z0-9]{6,})\b", re.IGNORECASE)

_ENGLISH_CODES = {"en-IN", "en-US", "en-GB", "en"}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    order_id: Optional[str] = None
    image_path: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    intent: str
    emotion: str
    language: str
    resolved: bool
    offer_given: Optional[dict] = None
    fraud_flagged: bool
    hallucination_flagged: bool
    tools_called: list
    latency_ms: float


class ImageUploadResponse(BaseModel):
    image_path: str
    session_id: str
    uploads_remaining: int


# ---------------------------------------------------------------------------
# NLP pipeline helpers (lazy imports — keep startup fast)
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> str:
    try:
        from nlp.language_detector import detect_language
        return detect_language(text)
    except Exception as exc:
        logger.warning("Language detection failed: %s", exc)
        return "en-IN"


def _analyze_emotion(text: str) -> str:
    try:
        from nlp.emotion_analyzer import analyze_emotion
        result = analyze_emotion(text)
        return result.get("label", "neutral")
    except Exception as exc:
        logger.warning("Emotion analysis failed: %s", exc)
        return "neutral"


def _translate_to_english(text: str, language: str) -> str:
    if language in _ENGLISH_CODES:
        return text
    try:
        from nlp.translator import translate_to_english
        return translate_to_english(text, language)
    except Exception as exc:
        logger.warning("Translation to English failed: %s", exc)
        return text


def _translate_to_language(text: str, language: str) -> str:
    if language in _ENGLISH_CODES:
        return text
    try:
        from nlp.translator import translate_to_language
        return translate_to_language(text, language)
    except Exception as exc:
        logger.warning("Translation to %s failed: %s", language, exc)
        return text


def _extract_order_id(text: str) -> Optional[str]:
    match = _ORDER_RE.search(text)
    return match.group(0).upper() if match else None


def _run_supervisor(
    user_id: str,
    session_id: str,
    message: str,
    language: str,
    emotion: str,
    order_id: Optional[str],
    image_path: Optional[str],
) -> dict:
    from agents.supervisor import run
    return run(
        user_id=user_id,
        session_id=session_id,
        message=message,
        language=language,
        emotion=emotion,
        order_id=order_id,
        image_path=image_path,
    )


def _record_metrics(result: dict, language: str, latency_s: float) -> None:
    try:
        from observability.metrics import (
            CHAT_REQUESTS, CHAT_LATENCY, HALLUCINATIONS,
            FRAUD_FLAGS, OFFERS_GIVEN, TOOL_CALLS,
        )
        intent = result.get("intent", "general")
        CHAT_REQUESTS.labels(intent=intent, language=language).inc()
        CHAT_LATENCY.labels(intent=intent).observe(latency_s)
        if result.get("hallucination_flagged"):
            HALLUCINATIONS.inc()
        if result.get("fraud_flagged"):
            FRAUD_FLAGS.inc()
        offer = result.get("offer_given")
        if offer:
            OFFERS_GIVEN.labels(offer_type=offer.get("offer_type", "unknown")).inc()
        for tool in result.get("tools_called", []):
            TOOL_CALLS.labels(tool_name=tool).inc()
    except Exception as exc:
        logger.debug("Metrics recording failed: %s", exc)


# ---------------------------------------------------------------------------
# Core processing function (shared by HTTP and WebSocket)
# ---------------------------------------------------------------------------

def _process_message(
    user_id: str,
    session_id: str,
    raw_message: str,
    order_id: Optional[str] = None,
    image_path: Optional[str] = None,
) -> tuple[dict, float]:
    """Run the full NLP → supervisor pipeline. Returns (result_dict, latency_ms)."""
    start = time.perf_counter()

    language = _detect_language(raw_message)
    emotion = _analyze_emotion(raw_message)
    english_message = _translate_to_english(raw_message, language)

    if not order_id:
        order_id = _extract_order_id(raw_message)

    result = _run_supervisor(
        user_id=user_id,
        session_id=session_id,
        message=english_message,
        language=language,
        emotion=emotion,
        order_id=order_id,
        image_path=image_path,
    )

    # Translate response back to customer language
    result["response"] = _translate_to_language(result.get("response", ""), language)
    result["language"] = language
    result["emotion"] = emotion

    latency_s = time.perf_counter() - start
    _record_metrics(result, language, latency_s)

    return result, round(latency_s * 1000, 2)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    from config import settings
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "timestamp": time.time(),
    }


@app.get("/")
async def root():
    return {
        "message": "Druta Kart AI Support API is running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics in text/plain format."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return JSONResponse(
            {"error": "prometheus_client not installed"},
            status_code=503,
        )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatRequest):
    """Main chat endpoint.

    Runs the full NLP pipeline → LangGraph supervisor → response translation.
    Rate-limited to 30 requests per minute per IP.
    """
    result, latency_ms = _process_message(
        user_id=body.user_id,
        session_id=body.session_id,
        raw_message=body.message,
        order_id=body.order_id,
        image_path=body.image_path,
    )

    return ChatResponse(
        response=result.get("response", ""),
        intent=result.get("intent", "general"),
        emotion=result.get("emotion", "neutral"),
        language=result.get("language", "en-IN"),
        resolved=result.get("resolved", False),
        offer_given=result.get("offer_given"),
        fraud_flagged=result.get("fraud_flagged", False),
        hallucination_flagged=result.get("hallucination_flagged", False),
        tools_called=result.get("tools_called", []),
        latency_ms=latency_ms,
    )


@app.post("/upload-image", response_model=ImageUploadResponse)
@limiter.limit("30/minute")
async def upload_image(
    request: Request,
    session_id: str,
    file: UploadFile = File(...),
):
    """Upload a complaint image.

    Limited to 3 uploads per session (tracked in-memory).
    Accepts JPEG, PNG, WEBP, GIF up to 10 MB.
    Returns the local file path to pass to subsequent /chat calls.
    """
    from config import settings

    # Session rate limit
    count = _session_image_counts.get(session_id, 0)
    if count >= settings.max_image_uploads_per_session:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Maximum {settings.max_image_uploads_per_session} image uploads "
                f"allowed per session."
            ),
        )

    # Validate content type
    content_type = file.content_type or ""
    if content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type: {content_type}. "
                   f"Allowed: {sorted(_ALLOWED_IMAGE_TYPES)}",
        )

    # Read and size-check
    data = await file.read()
    if len(data) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({len(data) // 1024} KB). Maximum 10 MB.",
        )

    # Save to temp directory with a unique name
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    filename = f"{session_id}_{uuid.uuid4().hex[:8]}{suffix}"
    dest = _UPLOAD_DIR / filename
    dest.write_bytes(data)

    _session_image_counts[session_id] = count + 1
    uploads_remaining = settings.max_image_uploads_per_session - (count + 1)

    try:
        from observability.metrics import IMAGE_UPLOADS
        IMAGE_UPLOADS.inc()
    except Exception:
        pass

    logger.info(
        "Image uploaded: session=%s file=%s size=%d bytes",
        session_id, filename, len(data),
    )

    return ImageUploadResponse(
        image_path=str(dest),
        session_id=session_id,
        uploads_remaining=max(uploads_remaining, 0),
    )


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket chat endpoint.

    Accepts JSON messages with shape:
        {"user_id": "...", "message": "...", "order_id": "..." (optional)}

    Sends JSON responses with the same shape as ChatResponse.
    """
    await websocket.accept()

    try:
        from observability.metrics import WS_CONNECTIONS
        WS_CONNECTIONS.inc()
    except Exception:
        pass

    logger.info("WebSocket connected: session=%s", session_id)

    try:
        while True:
            try:
                data = await websocket.receive_json()
            except Exception:
                break

            user_id = data.get("user_id", "anonymous")
            message = data.get("message", "").strip()
            order_id = data.get("order_id")
            image_path = data.get("image_path")

            if not message:
                await websocket.send_json({"error": "Empty message"})
                continue

            try:
                result, latency_ms = _process_message(
                    user_id=user_id,
                    session_id=session_id,
                    raw_message=message,
                    order_id=order_id,
                    image_path=image_path,
                )
                await websocket.send_json({
                    "response": result.get("response", ""),
                    "intent": result.get("intent", "general"),
                    "emotion": result.get("emotion", "neutral"),
                    "language": result.get("language", "en-IN"),
                    "resolved": result.get("resolved", False),
                    "offer_given": result.get("offer_given"),
                    "fraud_flagged": result.get("fraud_flagged", False),
                    "hallucination_flagged": result.get("hallucination_flagged", False),
                    "tools_called": result.get("tools_called", []),
                    "latency_ms": latency_ms,
                })
            except Exception as exc:
                logger.error("WebSocket handler error for session=%s: %s", session_id, exc)
                await websocket.send_json({
                    "error": "An internal error occurred. Please try again."
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session=%s", session_id)
    finally:
        try:
            from observability.metrics import WS_CONNECTIONS
            WS_CONNECTIONS.dec()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    from config import settings
    logger.info("=" * 50)
    logger.info("Druta Kart Backend starting...")
    logger.info("App:    %s v%s", settings.app_name, settings.app_version)
    logger.info("Debug:  %s", settings.debug)
    logger.info("Model:  %s", settings.groq_text_model)
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    # Clean up temp uploads on graceful shutdown
    try:
        shutil.rmtree(_UPLOAD_DIR, ignore_errors=True)
        logger.info("Cleaned up upload directory: %s", _UPLOAD_DIR)
    except Exception:
        pass
