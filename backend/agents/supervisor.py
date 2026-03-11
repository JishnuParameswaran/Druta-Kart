"""
Druta Kart - Supervisor Agent (LangGraph StateGraph).

Orchestrates all specialist agents in a directed graph:

  START
    └─► detect_intent
           ├─► complaint  ─► retention ─► respond ─► END
           ├─► order ──────────────────► respond ─► END
           ├─► dispatch ───────────────► respond ─► END
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
    vision_reason: Optional[str]        # Why Vision API classified the image (shown to customer)

    # Resolution
    resolved: bool
    resolution_type: Optional[str]
    csat_score: Optional[int]           # Internally set to 5 on resolution

    # Retention
    offer_given: Optional[dict]

    # Fraud
    fraud_flagged: bool

    # Human handoff (misidentification dispute escalation)
    human_handoff: bool                 # True when escalated to human agent
    handoff_proof: Optional[dict]       # {image_path, vision_reason, customer_messages}

    # Observability
    tools_called: List[str]
    agent_used: Optional[str]       # Which specialist agent handled this turn
    rag_used: bool                  # Whether RAG knowledge base was queried
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
            "Classify this customer support message into ONE category.\n"
            "The message may be in any Indian language (Hindi, Kannada, Tamil, Telugu, Bengali, Hinglish, Manglish, Kanglish, etc.) — classify by MEANING, not by language.\n\n"
            "Categories:\n"
            "- complaint  (customer RECEIVED the order but item is damaged, wrong item sent, missing item, expired food, quality issue)\n"
            "- order_tracking  (asking where is my order, what is ETA, requesting cancellation — item NOT yet received)\n"
            "- delivery  (not delivered at all, wrong address, delivery partner unreachable, refused delivery)\n"
            "- late_delivery  (order is overdue, taking too long, delayed, still waiting, order hasn't arrived but expected, any word meaning 'delay' or 'late' about an order)\n"
            "- payment  (payment failed, double charge, incorrect charge, refund status)\n"
            "- general  (happy customer, thank you, compliment, general product question, store hours, greeting, anything else)\n\n"
            "Critical rules:\n"
            "- If customer RECEIVED the order but something is wrong with it → complaint\n"
            "- Mentioning an order ID alone does NOT make it order_tracking — look at the actual problem\n"
            "- If customer is waiting and order has not arrived yet, OR uses any word meaning delay/late/overdue → late_delivery\n"
            "- Key delay words in Indian languages: ವಿಳಂಬ/ತಡ (Kannada), विलंब/देरी/नहीं आया (Hindi), தாமதம் (Tamil), ఆలస్యం (Telugu)\n"
            "- If unsure between complaint and delivery → complaint\n\n"
            "Examples (any language):\n"
            "- 'ನನ್ನ ಆರ್ಡರ್ ತಡವಾಗಿದೆ' (Kannada: my order is late) → late_delivery\n"
            "- 'ಸಮಯ ವಿಳಂಬ' or 'ವಿಳಂಬ' (Kannada: time delay / delay) → late_delivery\n"
            "- 'ಆರ್ಡರ್ ಬಂದಿಲ್ಲ' (Kannada: order hasn't come) → late_delivery\n"
            "- 'मेरा ऑर्डर अभी तक नहीं आया' (Hindi: order not arrived yet) → late_delivery\n"
            "- 'order bahut late ho raha hai' (Hinglish: order is very late) → late_delivery\n"
            "- 'wrong item received' → complaint\n"
            "- 'item missing from my order' → complaint\n"
            "- 'package is damaged' → complaint\n"
            "- 'where is my order' → order_tracking\n"
            "- 'payment failed but money deducted' → payment\n"
            f"Message: \"{user_message}\"\n\n"
            "Reply with ONLY the category word."
        ))])
        _tokens = result.content.strip().lower().split()
        intent = _tokens[0] if _tokens else "general"
        if intent not in ("complaint", "order_tracking", "delivery", "late_delivery", "payment", "general"):
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
    try:
        from agents.complaint_agent import run
        result = run(state)
        result["agent_used"] = "complaint_agent"
        return result
    except Exception as exc:
        logger.error("complaint_agent crashed: %s", exc)
        return {
            "response": (
                "We sincerely apologise for the difficulty with your complaint. "
                "Our team has been alerted and will follow up with you shortly."
            ),
            "resolved": False,
            "tools_called": state.get("tools_called", []),
            "agent_used": "complaint_agent",
        }


# ---------------------------------------------------------------------------
# Node: order
# ---------------------------------------------------------------------------

def order_node(state: AgentState) -> dict:
    try:
        from agents.order_agent import run
        result = run(state)
        result["agent_used"] = "order_agent"
        return result
    except Exception as exc:
        logger.error("order_agent crashed: %s", exc)
        return {
            "response": (
                "I'm having trouble retrieving your order details right now. "
                "Please try again in a moment or contact our support team."
            ),
            "resolved": False,
            "tools_called": state.get("tools_called", []),
            "agent_used": "order_agent",
        }


# ---------------------------------------------------------------------------
# Node: dispatch
# ---------------------------------------------------------------------------

def dispatch_node(state: AgentState) -> dict:
    try:
        from agents.dispatch_agent import run
        result = run(state)
        result["agent_used"] = "dispatch_agent"
        return result
    except Exception as exc:
        logger.error("dispatch_agent crashed: %s", exc)
        return {
            "response": (
                "We're sorry for the delivery issue. Our dispatch team has been "
                "alerted and will look into this immediately."
            ),
            "resolved": False,
            "tools_called": state.get("tools_called", []),
            "agent_used": "dispatch_agent",
        }


# ---------------------------------------------------------------------------
# Node: retention
# ---------------------------------------------------------------------------

def retention_node(state: AgentState) -> dict:
    # Only run retention if the complaint was actually resolved
    if not state.get("resolved"):
        return {}
    try:
        from agents.retention_agent import run
        return run(state)
    except Exception as exc:
        logger.error("retention_agent crashed: %s", exc)
        return {}


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
            "agent_used": "general",
        }
    except Exception as exc:
        logger.error("General node failed: %s", exc)
        return {
            "response": (
                "Thank you for contacting Druta Kart support. "
                "How can I help you today?"
            ),
            "resolved": False,
            "agent_used": "general",
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
    elif intent == "late_delivery":
        return "dispatch"
    elif intent == "delivery":
        return "dispatch"
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
        g.add_node("dispatch", dispatch_node)
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
                "dispatch": "dispatch",
                "general": "general",
            },
        )

        g.add_edge("complaint", "retention")
        g.add_edge("retention", "respond")
        g.add_edge("order", "respond")
        g.add_edge("dispatch", "respond")
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
        "vision_reason": None,
        "resolved": False,
        "resolution_type": None,
        "csat_score": None,
        "offer_given": None,
        "fraud_flagged": False,
        "human_handoff": False,
        "handoff_proof": None,
        "tools_called": [],
        "agent_used": None,
        "rag_used": False,
        "response": "",
        "hallucination_flagged": False,
    }

    try:
        result = graph.invoke(initial_state)

        # Repeated dispute after resolution: customer keeps messaging once bot has resolved
        _messages = result.get("messages", [])
        if len(_messages) >= 4 and result.get("resolved") is True:
            try:
                from agents.fraud_escalation_agent import run as _escalate
                _fraud_result = _escalate({
                    **initial_state,
                    **result,
                    "fraud_reason": "repeated_dispute_after_resolution",
                })
                result = {**result, **_fraud_result}
                logger.warning(
                    "Repeated dispute after resolution: user=%s session=%s messages=%d",
                    user_id, session_id, len(_messages),
                )
            except Exception as _exc:
                logger.error("Fraud escalation (repeated_dispute) failed: %s", _exc)

            try:
                from db.supabase_client import get_client
                _summary = " | ".join(
                    getattr(_m, "content", str(_m))[:120] for _m in _messages
                )
                get_client().table("security_events").insert({
                    "user_id": user_id,
                    "session_id": session_id,
                    "event_type": "repeated_dispute_after_resolution",
                    "summary": _summary,
                    "message_count": len(_messages),
                }).execute()
            except Exception as _exc:
                logger.warning("security_events logging failed: %s", _exc)

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
            "human_handoff": result.get("human_handoff", False),
            "csat_score": result.get("csat_score"),
            "handoff_proof": result.get("handoff_proof"),
            "agent_used": result.get("agent_used"),
            "rag_used": "rag_search" in result.get("tools_called", []),
            "image_validation_result": result.get("image_validation_result"),
        }
    except Exception as exc:
        logger.error("Supervisor graph.invoke failed: %s", exc, exc_info=True)
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
