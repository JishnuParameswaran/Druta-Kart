"""
Druta Kart - Prometheus Metrics

Counters and histograms exposed at /metrics via prometheus_client.
Imported by main.py and optionally by individual agents for fine-grained tracking.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Histogram, Gauge

    # Chat volume
    CHAT_REQUESTS = Counter(
        "druta_kart_chat_requests_total",
        "Total chat requests received",
        ["intent", "language"],
    )

    # Latency
    CHAT_LATENCY = Histogram(
        "druta_kart_chat_latency_seconds",
        "End-to-end chat response latency in seconds",
        ["intent"],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    )

    # Quality signals
    HALLUCINATIONS = Counter(
        "druta_kart_hallucinations_total",
        "Number of responses flagged by the hallucination guard",
    )

    FRAUD_FLAGS = Counter(
        "druta_kart_fraud_flags_total",
        "Number of fraud escalation events triggered",
    )

    # Retention
    OFFERS_GIVEN = Counter(
        "druta_kart_offers_given_total",
        "Retention offers presented to customers",
        ["offer_type"],
    )

    # Tool usage
    TOOL_CALLS = Counter(
        "druta_kart_tool_calls_total",
        "LangChain tool invocations",
        ["tool_name"],
    )

    # Active WebSocket connections
    WS_CONNECTIONS = Gauge(
        "druta_kart_ws_connections_active",
        "Number of active WebSocket connections",
    )

    # Image uploads
    IMAGE_UPLOADS = Counter(
        "druta_kart_image_uploads_total",
        "Complaint image uploads received",
    )

    _METRICS_AVAILABLE = True
    logger.info("Prometheus metrics initialised.")

except ImportError:
    logger.warning(
        "prometheus_client not installed; metrics will be no-ops. "
        "Install with: pip install prometheus-client"
    )
    _METRICS_AVAILABLE = False

    # Provide no-op stubs so callers don't need to guard every metric call
    class _Noop:
        def labels(self, **_):
            return self
        def inc(self, *_, **__):
            pass
        def observe(self, *_, **__):
            pass
        def set(self, *_, **__):
            pass
        def time(self):
            import contextlib
            return contextlib.nullcontext()

    CHAT_REQUESTS = _Noop()
    CHAT_LATENCY = _Noop()
    HALLUCINATIONS = _Noop()
    FRAUD_FLAGS = _Noop()
    OFFERS_GIVEN = _Noop()
    TOOL_CALLS = _Noop()
    WS_CONNECTIONS = _Noop()
    IMAGE_UPLOADS = _Noop()


def metrics_available() -> bool:
    return _METRICS_AVAILABLE
