"""
Druta Kart - Supervisor Agent (LangGraph StateGraph).

Orchestrates all specialist agents in a directed graph:

  START
    └─► detect_intent
           ├─► complaint  ─► retention ─► respond ─► END
           ├─► order ──────────────────► respond ─► END
           └─► general ─────────────────► respond ─► END

State flows through nodes as a typed dict.  Each node function receives the
full state and returns a partial dict with only the fields it modifies.
"""
from __future__ import annotations

import logging
import operator
from typing import Annotated, List, Optional, TypedDict

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # Identity
    user_id: str
    session_id: str

    # NLP context (set by main.py NLP pipeline before supervisor runs)
    language: str           # e.g. "en-IN", "hi-IN"
    emotion: str            # e.g. "anger", "neutral"

    # Conversation history — LangGraph appends with operator.add
    messages: Annotated[List, operator.add]

    # Routing
    intent: str             # "complaint" | "order_tracking" | "payment" | "general"

    # Complaint specifics
    complaint_type: Optional[str]   # "damaged" | "missing" | "wrong" | "expired" | "payment"
    order_id: Optional[str]

    # Image validation
    image_path: Optional[str]
    image_validation_result: Optional[str]

    # Resolution
    resolved: bool
    resolution_type: Optional[str]

    # Retention
    offer_given: Optional[dict]

    # Fraud
    fraud_flagged: bool

    # Observability
    tools_called: List[str]
    response: str
    hallucination_flagged: bool


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=settings.groq_text_model,
        api_key=settings.groq_api_key,
        temperature=0.2,
    )


def _get_system_prompt() -> str:
    try:
        from prompts.prompt_registry import get_prompt
        return get_prompt("system_prompt")
    except Exception:
        return "You are a helpful customer support agent for Druta Kart."


def _last_user_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        if role in ("human", "user"):
            return getattr(msg, "content", str(msg))
    return ""


# ---------------------------------------------------------------------------
# Node: detect_intent
# ---------------------------------------------------------------------------

def detect_intent_node(state: AgentState) -> dict:
    """Classify the customer's intent from the latest message."""
    user_message = _last_user_message(state)
    if not user_message:
        return {"intent": "general"}

    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage
        result = llm.invoke([HumanMessage(content=(
            "Classify this customer support message into ONE category:\n"
            "- complaint  (damaged item, missing item, wrong item, expired food)\n"
            "- order_tracking  (where is my order, late delivery, ETA, cancellation)\n"
            "- payment  (payment failed, double charge, refund status)\n"
            "- general  (anything else)\n\n"
            f"Message: \"{user_message}\"\n\n"
            "Reply with ONLY the category word."
        ))])
        intent = result.content.strip().lower().split()[0]
        if intent not in ("complaint", "order_tracking", "payment", "general"):
            intent = "general"
        logger.info("Intent detected: %s for user=%s", intent, state.get("user_id"))
        return {"intent": intent}
    except Exception as exc:
        logger.error("Intent detection failed: %s", exc)
        return {"intent": "general"}


# ---------------------------------------------------------------------------
# Node: complaint
# ---------------------------------------------------------------------------

def complaint_node(state: AgentState) -> dict:
    from agents.complaint_agent import run
    return run(state)


# ---------------------------------------------------------------------------
# Node: order
# ---------------------------------------------------------------------------

def order_node(state: AgentState) -> dict:
    from agents.order_agent import run
    return run(state)


# ---------------------------------------------------------------------------
# Node: retention
# ---------------------------------------------------------------------------

def retention_node(state: AgentState) -> dict:
    # Only run retention if the complaint was actually resolved
    if not state.get("resolved"):
        return {}
    from agents.retention_agent import run
    return run(state)


# ---------------------------------------------------------------------------
# Node: general
# ---------------------------------------------------------------------------

def general_node(state: AgentState) -> dict:
    """Handle general queries directly with the LLM + system prompt."""
    user_message = _last_user_message(state)
    language = state.get("language", "en-IN")
    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage, SystemMessage
        result = llm.invoke([
            SystemMessage(content=_get_system_prompt()),
            HumanMessage(content=(
                f"Customer message: \"{user_message}\"\n"
                f"Respond helpfully in {language}. "
                f"If you cannot resolve the issue, offer to connect them with a human agent."
            )),
        ])
        return {
            "response": result.content.strip(),
            "resolved": True,
        }
    except Exception as exc:
        logger.error("General node failed: %s", exc)
        return {
            "response": (
                "Thank you for contacting Druta Kart support. "
                "How can I help you today?"
            ),
            "resolved": False,
        }


# ---------------------------------------------------------------------------
# Node: respond (final — hallucination guard + logging)
# ---------------------------------------------------------------------------

def respond_node(state: AgentState) -> dict:
    """Apply hallucination guard and log the interaction."""
    response = state.get("response", "")
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")

    # Hallucination guard
    hallucination_flagged = False
    try:
        from agents.hallucination_guard import check_response
        guard_result = check_response(response, session_id=session_id, user_id=user_id)
        response = guard_result["response"]
        hallucination_flagged = guard_result["hallucination_flagged"]
    except Exception as exc:
        logger.warning("Hallucination guard error: %s", exc)

    # Log interaction
    try:
        from observability.logger import log_interaction
        import time
        log_interaction(
            session_id=session_id,
            user_id=user_id,
            user_message=_last_user_message(state),
            bot_response=response,
            language=state.get("language", "en-IN"),
            emotion=state.get("emotion", "neutral"),
            intent=state.get("intent", ""),
            agent_used=state.get("intent", "general"),
            tools_called=state.get("tools_called", []),
            offer_given=state.get("offer_given") or {},
            resolution_status="resolved" if state.get("resolved") else "unresolved",
            latency_ms=0,
            hallucination_flagged=hallucination_flagged,
        )
    except Exception as exc:
        logger.warning("Interaction logging failed: %s", exc)

    return {
        "response": response,
        "hallucination_flagged": hallucination_flagged,
    }


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def _route_intent(state: AgentState) -> str:
    intent = state.get("intent", "general")
    if intent in ("complaint", "payment"):
        return "complaint"
    elif intent == "order_tracking":
        return "order"
    else:
        return "general"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def _build_graph():
    try:
        from langgraph.graph import StateGraph, END

        g = StateGraph(AgentState)
        g.add_node("detect_intent", detect_intent_node)
        g.add_node("complaint", complaint_node)
        g.add_node("order", order_node)
        g.add_node("retention", retention_node)
        g.add_node("general", general_node)
        g.add_node("respond", respond_node)

        g.set_entry_point("detect_intent")

        g.add_conditional_edges(
            "detect_intent",
            _route_intent,
            {
                "complaint": "complaint",
                "order": "order",
                "general": "general",
            },
        )

        g.add_edge("complaint", "retention")
        g.add_edge("retention", "respond")
        g.add_edge("order", "respond")
        g.add_edge("general", "respond")
        g.add_edge("respond", END)

        return g.compile()

    except ImportError as exc:
        logger.warning("LangGraph not available (%s); graph will be None.", exc)
        return None


graph = _build_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    user_id: str,
    session_id: str,
    message: str,
    language: str = "en-IN",
    emotion: str = "neutral",
    order_id: Optional[str] = None,
    image_path: Optional[str] = None,
) -> dict:
    """Main entry point called by FastAPI routes and WebSocket handler.

    Args:
        user_id:    Customer user ID.
        session_id: Current chat session ID.
        message:    Latest customer message (already translated to English if needed).
        language:   Detected language code (e.g. "hi-IN") for response localisation.
        emotion:    Detected emotion string (e.g. "anger").
        order_id:   Order ID extracted from the message by main.py (optional).
        image_path: Local path to uploaded image file (optional).

    Returns:
        Dict with keys: response, intent, emotion, language, resolved,
        offer_given, fraud_flagged, hallucination_flagged, tools_called.
    """
    if graph is None:
        return {
            "response": "Support service is temporarily unavailable. Please try again.",
            "intent": "general",
            "emotion": emotion,
            "language": language,
            "resolved": False,
            "offer_given": None,
            "fraud_flagged": False,
            "hallucination_flagged": False,
            "tools_called": [],
        }

    try:
        from langchain_core.messages import HumanMessage
        human_msg = HumanMessage(content=message)
    except ImportError:
        # Fallback: use plain dict if langchain_core not installed
        human_msg = {"role": "user", "content": message}  # type: ignore

    initial_state: AgentState = {
        "user_id": user_id,
        "session_id": session_id,
        "language": language,
        "emotion": emotion,
        "messages": [human_msg],
        "intent": "",
        "complaint_type": None,
        "order_id": order_id,
        "image_path": image_path,
        "image_validation_result": None,
        "resolved": False,
        "resolution_type": None,
        "offer_given": None,
        "fraud_flagged": False,
        "tools_called": [],
        "response": "",
        "hallucination_flagged": False,
    }

    try:
        result = graph.invoke(initial_state)
        return {
            "response": result.get("response", ""),
            "intent": result.get("intent", "general"),
            "emotion": result.get("emotion", emotion),
            "language": result.get("language", language),
            "resolved": result.get("resolved", False),
            "offer_given": result.get("offer_given"),
            "fraud_flagged": result.get("fraud_flagged", False),
            "hallucination_flagged": result.get("hallucination_flagged", False),
            "tools_called": result.get("tools_called", []),
        }
    except Exception as exc:
        logger.error("Supervisor graph.invoke failed: %s", exc)
        return {
            "response": (
                "I'm sorry, I encountered an issue processing your request. "
                "Please try again or contact our support team."
            ),
            "intent": "general",
            "emotion": emotion,
            "language": language,
            "resolved": False,
            "offer_given": None,
            "fraud_flagged": False,
            "hallucination_flagged": False,
            "tools_called": [],
        }
