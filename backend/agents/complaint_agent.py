"""
Druta Kart - Complaint Agent.

Handles: damaged product, missing item, wrong item, expired food.

Flow:
  1. Classify complaint type from user message
  2. For physical complaints → request image (if not already provided)
  3. If image present → run image_validation_agent
     a. real_damage → continue to resolution
     b. ai_generated/suspicious → ask for live photo; fraud escalation if refused
     c. misidentification → educate, do not resolve
  4. Query RAG for relevant policy context
  5. Call appropriate tool (refund / replacement / wallet_credit)
  6. Call dispatch_checklist_tool
  7. Return resolved state (retention_agent picks up next)
"""
from __future__ import annotations

import logging
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

_PHYSICAL_COMPLAINT_TYPES = {"damaged", "wrong", "expired", "missing"}

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
        from langchain_core.messages import HumanMessage, SystemMessage

        # Determine best resolution from LLM given complaint type
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

    tool_result: dict = {}

    if resolution == "refund" and order_id:
        try:
            from tools.refund_tool import refund_tool
            tool_result = refund_tool.invoke({
                "order_id": order_id,
                "user_id": user_id,
                "amount_inr": 0.0,  # amount determined by finance team
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
            f"4. Confirms next steps\n"
            f"Keep it concise (3-4 sentences). Do NOT invent order IDs or dates."
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
            f"We have initiated a resolution and will follow up shortly."
        )

    return response, resolution, tools_called


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

    # Step 2 — Image required but not provided yet
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

    # Step 3 — Image validation
    if image_path and not image_validation_result:
        from agents.image_validation_agent import run as validate_image
        validation_update = validate_image(state)
        image_validation_result = validation_update.get("image_validation_result")
        tools_called = validation_update.get("tools_called", tools_called)

        # If not real damage, return the validation agent's response
        if image_validation_result != "real_damage":
            # Check if customer has refused live photo (ai_generated/suspicious + already asked)
            if image_validation_result in ("ai_generated", "suspicious"):
                # In this simple implementation, route to fraud escalation
                # In production, check session history for prior requests
                from agents.fraud_escalation_agent import run as escalate_fraud
                fraud_update = escalate_fraud({**state, **validation_update})
                return {
                    **fraud_update,
                    "complaint_type": complaint_type,
                    "image_validation_result": image_validation_result,
                }
            return {
                **validation_update,
                "complaint_type": complaint_type,
                "resolved": False,
            }

    # Step 4 — Query RAG for policy context
    policy_context = _query_policy(
        f"{complaint_type} product refund replacement policy"
    )

    # Step 5 — Resolve complaint
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

    # Step 6 — Dispatch checklist
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

    return {
        "complaint_type": complaint_type,
        "response": response,
        "resolution_type": resolution_type,
        "resolved": True,
        "image_validation_result": image_validation_result,
        "tools_called": tools_called,
    }
