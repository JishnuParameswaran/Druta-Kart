"""
Druta Kart - Structured JSON Logger
"""
import structlog
import logging
import time

logging.basicConfig(level=logging.INFO)

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()


def log_interaction(
    session_id: str,
    user_id: str,
    user_message: str,
    bot_response: str,
    language: str,
    emotion: str,
    intent: str,
    agent_used: str,
    tools_called: list,
    offer_given: dict,
    resolution_status: str,
    latency_ms: float,
    token_usage: dict = None,
    hallucination_flagged: bool = False
):
    logger.info(
        "interaction",
        session_id=session_id,
        user_id=user_id,
        user_message=user_message[:200],
        bot_response=bot_response[:200],
        language=language,
        emotion=emotion,
        intent=intent,
        agent_used=agent_used,
        tools_called=tools_called,
        offer_given=offer_given,
        resolution_status=resolution_status,
        latency_ms=latency_ms,
        token_usage=token_usage or {},
        hallucination_flagged=hallucination_flagged,
        timestamp=time.time()
    )