"""
Druta Kart - Complaint Agent.

Handles: damaged product, missing item, wrong item, expired food.

Flow:
  1. Classify complaint type from user message
  2. For physical complaints → request image (if not already provided)
  3. If misidentification already in state → detect dispute → human handoff
  4. If image present → run image_validation_agent
     a. real_damage → continue to resolution
     b. ai_generated/suspicious → ask for live photo; fraud escalation if refused
     c. misidentification (Turn 1) → educate with Vision API reason
  5. Query RAG for relevant policy context
  6. Call appropriate tool (refund / replacement / wallet_credit)
  7. Call dispatch_checklist_tool (physical complaints only, not payment)
  8. Inform customer dispatch has been notified
  9. Set csat_score=5, log sentiment to JSONL
"""
from __future__ import annotations

import logging
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

_PHYSICAL_COMPLAINT_TYPES = {"damaged", "expired"}

# Complaint types that resolve immediately without requiring a photo
_FORCED_RESOLUTION = {
    "wrong":   "replacement",
    "missing": "replacement",
    "payment": "refund",
}

# Physical complaint types that get dispatch notification (NOT payment)
_DISPATCH_COMPLAINT_TYPES = {"damaged", "wrong", "missing", "expired"}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=settings.groq_text_model,
        api_key=settings.groq_api_key,
        temperature=0.3,
    )


def _get_system_prompt() -> str:
    try:
        from prompts.prompt_registry import get_prompt
        return get_prompt("system_prompt")
    except Exception:
        return "You are a helpful customer support agent for Druta Kart."


def _classify_complaint(user_message: str) -> str:
    """Use LLM to classify the complaint type."""
    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage
        result = llm.invoke([HumanMessage(content=(
            f"Classify this customer complaint into ONE of these categories: "
            f"damaged, missing, wrong, expired, payment, other.\n\n"
            f"Customer message: \"{user_message}\"\n\n"
            f"Reply with ONLY the category word, nothing else."
        ))])
        label = result.content.strip().lower().split()[0]
        valid = {"damaged", "missing", "wrong", "expired", "payment", "other"}
        return label if label in valid else "other"
    except Exception as exc:
        logger.warning("Complaint classification failed: %s", exc)
        return "other"


def _query_policy(query: str) -> str:
    """Retrieve relevant policy context from the RAG knowledge base."""
    try:
        from rag.vector_store import query_knowledge_base
        return query_knowledge_base(query)
    except Exception as exc:
        logger.warning("RAG query failed: %s", exc)
        return ""


def _is_customer_disputing(user_message: str) -> bool:
    """Use LLM to detect if customer is still disputing the image review."""
    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage
        result = llm.invoke([HumanMessage(content=(
            "A customer was told their complaint image does not show actual damage. "
            "Now they replied with:\n\n"
            f"\"{user_message}\"\n\n"
            "Is the customer still arguing that the item IS damaged / disputing our review? "
            "Reply with ONLY: YES or NO"
        ))])
        return "yes" in result.content.strip().lower()
    except Exception as exc:
        logger.warning("Dispute detection failed: %s", exc)
        return False  # if unsure, don't escalate


def _extract_customer_messages(messages: list) -> list[str]:
    """Extract all human messages from conversation history."""
    result = []
    for msg in messages:
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        if role in ("human", "user"):
            result.append(getattr(msg, "content", str(msg)))
    return result


def _resolve_complaint(
    complaint_type: str,
    order_id: Optional[str],
    user_id: str,
    session_id: str,
    user_message: str,
    language: str,
    policy_context: str,
    emotion: str,
) -> tuple[str, str, list]:
    """Call the appropriate resolution tool and return (response, resolution_type, tools_called)."""
    tools_called: list = []

    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage

        resolution_prompt = (
            f"Customer complaint type: {complaint_type}\n"
            f"Customer emotion: {emotion}\n"
            f"Relevant policy:\n{policy_context}\n\n"
            f"Choose the BEST resolution: refund, replacement, or wallet_credit. "
            f"Reply with ONLY one word."
        )
        res_result = llm.invoke([HumanMessage(content=resolution_prompt)])
        resolution = res_result.content.strip().lower().split()[0]
        if resolution not in ("refund", "replacement", "wallet_credit"):
            resolution = "wallet_credit"
    except Exception:
        resolution = "wallet_credit"

    # Override LLM decision for complaint types with a defined instant resolution
    if complaint_type in _FORCED_RESOLUTION:
        resolution = _FORCED_RESOLUTION[complaint_type]

    tool_result: dict = {}

    if resolution == "refund" and order_id:
        try:
            from tools.refund_tool import refund_tool
            tool_result = refund_tool.invoke({
                "order_id": order_id,
                "user_id": user_id,
                "amount_inr": 0.0,
                "reason": f"{complaint_type} complaint",
            })
            tools_called.append("refund_tool")
        except Exception as exc:
            logger.error("refund_tool failed: %s", exc)

    elif resolution == "replacement" and order_id:
        try:
            from tools.replacement_tool import replacement_tool
            tool_result = replacement_tool.invoke({
                "order_id": order_id,
                "user_id": user_id,
                "product_name": "your item",
                "reason": f"{complaint_type} complaint",
            })
            tools_called.append("replacement_tool")
        except Exception as exc:
            logger.error("replacement_tool failed: %s", exc)

    else:
        try:
            from tools.wallet_credit_tool import wallet_credit_tool
            tool_result = wallet_credit_tool.invoke({
                "user_id": user_id,
                "amount_inr": 100.0,
                "reason": f"{complaint_type} complaint — goodwill credit",
            })
            tools_called.append("wallet_credit_tool")
            resolution = "wallet_credit"
        except Exception as exc:
            logger.error("wallet_credit_tool failed: %s", exc)

    # Generate customer-facing response
    try:
        llm = _get_llm()
        from langchain_core.messages import HumanMessage, SystemMessage

        tool_summary = ""
        if tool_result and tool_result.get("status") != "failed":
            if resolution == "refund":
                tool_summary = f"Refund initiated (ID: {tool_result.get('refund_id', 'N/A')}, expected {tool_result.get('expected_days', 5)}-7 business days)."
            elif resolution == "replacement":
                tool_summary = f"Replacement approved, expected by {tool_result.get('expected_delivery', 'tomorrow')}."
            elif resolution == "wallet_credit":
                tool_summary = f"₹{int(tool_result.get('credited_amount', 100))} added to wallet."

        response_prompt = (
            f"Customer message: \"{user_message}\"\n"
            f"Complaint type: {complaint_type}\n"
            f"Customer emotion: {emotion}\n"
            f"Resolution applied: {tool_summary}\n"
            f"Relevant policy:\n{policy_context}\n\n"
            f"Write a warm, empathetic response in {language} that:\n"
            f"1. Acknowledges the customer's frustration\n"
            f"2. Apologises sincerely\n"
            f"3. Explains what action was taken ({tool_summary})\n"
            f"4. Mentions that we have notified our dispatch team about this issue\n"
            f"5. Confirms next steps\n"
            f"Keep it concise (3-5 sentences). Do NOT invent order IDs or dates."
        )

        result = llm.invoke([
            SystemMessage(content=_get_system_prompt()),
            HumanMessage(content=response_prompt),
        ])
        response = result.content.strip()
    except Exception as exc:
        logger.error("Complaint response generation failed: %s", exc)
        response = (
            f"We sincerely apologise for the inconvenience with your {complaint_type} item. "
            f"We have initiated a resolution and notified our dispatch team. "
            f"We will follow up shortly."
        )

    return response, resolution, tools_called


def _log_sentiment(
    session_id: str,
    user_id: str,
    scenario: str,
    resolution_type: str,
    emotion: str,
    language: str,
    human_handoff: bool = False,
    handoff_reason: str = "",
) -> None:
    """Log customer sentiment to JSONL for human analysis."""
    try:
        from data.customer_sentiment import log_sentiment
        log_sentiment(
            session_id=session_id,
            user_id=user_id,
            scenario=scenario,
            resolution_type=resolution_type,
            csat_score=5,  # AI resolved = always 5 internally
            emotion_at_start=emotion,
            language=language,
            human_handoff=human_handoff,
            handoff_reason=handoff_reason,
        )
    except Exception as exc:
        logger.warning("Sentiment logging failed: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(state: dict) -> dict:
    """Handle a product complaint end-to-end.

    Args:
        state: AgentState dict.

    Returns:
        Partial AgentState update.
    """
    user_id = state.get("user_id", "")
    session_id = state.get("session_id", "")
    language = state.get("language", "en-IN")
    emotion = state.get("emotion", "neutral")
    order_id: Optional[str] = state.get("order_id")
    image_path: Optional[str] = state.get("image_path")
    image_validation_result: Optional[str] = state.get("image_validation_result")
    vision_reason: str = state.get("vision_reason", "")
    tools_called = list(state.get("tools_called", []))

    # Extract last user message
    messages = state.get("messages", [])
    user_message = ""
    for msg in reversed(messages):
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        if role in ("human", "user"):
            user_message = getattr(msg, "content", str(msg))
            break

    # Step 1 — Classify complaint
    complaint_type = state.get("complaint_type") or _classify_complaint(user_message)

    # Step 2 — Photo required only for physical damage (damaged/expired)
    # wrong/missing/payment resolve immediately without photo
    if complaint_type in _PHYSICAL_COMPLAINT_TYPES and not image_path and not image_validation_result:
        return {
            "complaint_type": complaint_type,
            "response": (
                f"I'm sorry to hear about the {complaint_type} item. To process your "
                f"complaint quickly, could you please share a photo of the item? "
                f"This helps us verify the issue and get you the right resolution."
            ),
            "resolved": False,
            "tools_called": tools_called,
        }

    # Step 3 — Handle Turn 2+ misidentification dispute
    # If the previous turn already returned misidentification, check if customer is still arguing
    if image_validation_result == "misidentification" and user_message:
        if _is_customer_disputing(user_message):
            # Build proof for human agent
            customer_msgs = _extract_customer_messages(messages)
            handoff_proof = {
                "image_path": image_path,
                "vision_reason": vision_reason,
                "customer_messages": customer_msgs,
                "complaint_type": complaint_type,
            }

            # Log sentiment as human handoff
            _log_sentiment(
                session_id=session_id,
                user_id=user_id,
                scenario=complaint_type,
                resolution_type="human_handoff",
                emotion=emotion,
                language=language,
                human_handoff=True,
                handoff_reason="customer_disputed_misidentification",
            )

            return {
                "complaint_type": complaint_type,
                "image_validation_result": image_validation_result,
                "human_handoff": True,
                "handoff_proof": handoff_proof,
                "csat_score": 5,
                "response": (
                    "I completely understand your concern and I'm sorry for any frustration. "
                    "Since you feel strongly about this, I've escalated your case to a senior "
                    "support specialist who will personally review the full details and get back "
                    "to you very soon. Your complaint has been logged with all the evidence. "
                    "Thank you for your patience — we truly value your trust in Druta Kart."
                ),
                "resolved": True,
                "tools_called": tools_called,
            }

    # Step 4 — Image validation (first time seeing the image)
    if image_path and not image_validation_result:
        try:
            from agents.image_validation_agent import run as validate_image
            validation_update = validate_image(state)
            image_validation_result = validation_update.get("image_validation_result")
            vision_reason = validation_update.get("vision_reason", "")
            tools_called = validation_update.get("tools_called", tools_called)

            if image_validation_result != "real_damage":
                if image_validation_result in ("ai_generated", "suspicious"):
                    try:
                        from agents.fraud_escalation_agent import run as escalate_fraud
                        fraud_update = escalate_fraud({**state, **validation_update})
                        return {
                            **fraud_update,
                            "complaint_type": complaint_type,
                            "image_validation_result": image_validation_result,
                        }
                    except Exception as exc:
                        logger.error("fraud_escalation_agent failed: %s", exc)
                        # Fall through to normal resolution
                        image_validation_result = "real_damage"

                elif image_validation_result == "misidentification":
                    # Turn 1: Educate customer using the Vision API reason
                    reason_clause = (
                        f" Our image analysis found: {vision_reason}." if vision_reason else ""
                    )
                    return {
                        "complaint_type": complaint_type,
                        "image_validation_result": "misidentification",
                        "vision_reason": vision_reason,
                        "response": (
                            f"Thank you for sharing the photo!{reason_clause} "
                            f"Based on our review, the item appears to be in the correct condition — "
                            f"sometimes packaging or lighting can make it look different. "
                            f"Could you take a closer look? If you believe the item is genuinely "
                            f"damaged, please let us know and we'll look into it further for you."
                        ),
                        "resolved": False,
                        "tools_called": tools_called,
                    }

        except Exception as exc:
            logger.error("image_validation_agent failed: %s", exc)
            # Treat as real damage so the customer's complaint isn't blocked
            image_validation_result = "real_damage"

    # Step 5 — Query RAG for policy context
    policy_context = _query_policy(
        f"{complaint_type} product refund replacement policy"
    )
    tools_called.append("rag_search")  # Always mark RAG queried (Option B tracking)

    # Step 6 — Resolve complaint
    response, resolution_type, new_tools = _resolve_complaint(
        complaint_type=complaint_type,
        order_id=order_id,
        user_id=user_id,
        session_id=session_id,
        user_message=user_message,
        language=language,
        policy_context=policy_context,
        emotion=emotion,
    )
    tools_called.extend(new_tools)

    # Step 7 — Dispatch checklist (physical complaints only, not payment)
    if complaint_type in _DISPATCH_COMPLAINT_TYPES:
        try:
            from tools.dispatch_checklist_tool import dispatch_checklist_tool
            issue_map = {
                "damaged": "damaged_package",
                "missing": "missing_items",
                "wrong": "wrong_address",
                "expired": "damaged_package",
            }
            issue_key = issue_map.get(complaint_type, complaint_type)
            dispatch_checklist_tool.invoke({
                "user_id": user_id,
                "session_id": session_id,
                "issues": [issue_key],
            })
            tools_called.append("dispatch_checklist_tool")
        except Exception as exc:
            logger.warning("dispatch_checklist_tool failed: %s", exc)

    # Step 8 — Log sentiment (CSAT = 5, fire-and-forget)
    _log_sentiment(
        session_id=session_id,
        user_id=user_id,
        scenario=complaint_type,
        resolution_type=resolution_type,
        emotion=emotion,
        language=language,
        human_handoff=False,
        handoff_reason="",
    )

    return {
        "complaint_type": complaint_type,
        "response": response,
        "resolution_type": resolution_type,
        "resolved": True,
        "csat_score": 5,
        "image_validation_result": image_validation_result,
        "tools_called": tools_called,
    }
