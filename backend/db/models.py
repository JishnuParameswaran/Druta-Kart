"""
Druta Kart - Pydantic data models for database entities.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CustomerSegment(str, Enum):
    new = "new"
    regular = "regular"
    bulk = "bulk"
    churning = "churning"
    frequent_complainer = "frequent_complainer"


class ComplaintType(str, Enum):
    damaged = "damaged"
    missing = "missing"
    wrong = "wrong"
    late = "late"
    payment = "payment"


class ResolutionType(str, Enum):
    refund = "refund"
    replacement = "replacement"
    wallet = "wallet"
    offer = "offer"
    none = "none"


class MessageRole(str, Enum):
    user = "user"
    bot = "bot"


class ResolutionStatus(str, Enum):
    resolved = "resolved"
    unresolved = "unresolved"
    escalated = "escalated"
    pending = "pending"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CustomerProfile(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    user_id: str
    name: str
    phone: Optional[str] = None
    total_orders: int = 0
    avg_spend_inr: float = 0.0
    complaint_count: int = 0
    last_complaint_date: Optional[datetime] = None
    satisfaction_score: Optional[float] = None  # 0.0 – 5.0
    customer_segment: CustomerSegment = CustomerSegment.new
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ChatSession(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    resolution_status: ResolutionStatus = ResolutionStatus.pending
    emotion_detected: Optional[str] = None   # e.g. "angry", "neutral"
    language_detected: Optional[str] = None  # e.g. "hi", "en", "ta"


class ChatMessage(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    message_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: str
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    emotion: Optional[str] = None
    language: Optional[str] = None
    agent_used: Optional[str] = None
    tools_called: List[str] = Field(default_factory=list)
    latency_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    hallucination_flagged: bool = False


class ComplaintLog(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    complaint_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: str
    complaint_type: ComplaintType
    product_name: Optional[str] = None
    resolution_type: ResolutionType = ResolutionType.none
    resolved_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class OfferGiven(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    offer_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    session_id: str
    offer_type: str           # e.g. "wallet_credit", "discount", "free_item"
    offer_value: float        # numeric value (INR or %)
    offer_description: str
    accepted: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DispatchChecklist(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    checklist_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: str
    issues_reported: List[str] = Field(default_factory=list)
    checklist_items: List[str] = Field(default_factory=list)
    sent_at: datetime = Field(default_factory=datetime.utcnow)
