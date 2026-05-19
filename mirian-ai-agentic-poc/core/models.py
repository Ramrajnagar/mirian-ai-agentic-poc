"""
core/models.py
--------------
Single source of truth for all financial data models used across the system.

Every model uses Pydantic v2 for strict validation. No raw dicts are passed between
modules — everything is typed. This prevents LLM hallucinations from corrupting
financial pipelines and makes every data flow auditable.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RiskTier(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class CommunicationTone(str, Enum):
    CONCIERGE = "Concierge"      # Prompt payer, minor slip
    COLLABORATIVE = "Collaborative"  # Moderate risk, first nudge
    FIRM = "Firm"                # Slow/repeat late payer
    LEGAL_WARNING = "LegalWarning"   # Chronic non-payer


class PaymentStatus(str, Enum):
    PAID = "paid"
    PENDING = "pending"
    OVERDUE = "overdue"
    DISPUTED = "disputed"
    WRITTEN_OFF = "written_off"


class EscalationChannel(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    COLLECTIONS_AGENCY = "collections_agency"
    LEGAL = "legal"


# ---------------------------------------------------------------------------
# Customer models
# ---------------------------------------------------------------------------


class PaymentRecord(BaseModel):
    """One historical payment event for a customer."""

    invoice_id: str
    invoice_amount: float = Field(gt=0)
    due_date: date
    paid_date: Optional[date] = None
    days_late: int = Field(default=0, ge=0)  # 0 = on time or early


class Customer(BaseModel):
    """Full customer profile with behavioral payment history."""

    customer_id: str
    company_name: str
    contact_name: str
    contact_email: str
    industry: str
    credit_limit: float = Field(gt=0)
    payment_history: list[PaymentRecord] = Field(default_factory=list)

    @computed_field  # type: ignore[misc]
    @property
    def average_days_late(self) -> float:
        """Average days late across all historical payments."""
        if not self.payment_history:
            return 0.0
        return sum(r.days_late for r in self.payment_history) / len(self.payment_history)

    @computed_field  # type: ignore[misc]
    @property
    def payment_count(self) -> int:
        return len(self.payment_history)

    @computed_field  # type: ignore[misc]
    @property
    def chronic_late_rate(self) -> float:
        """Fraction of payments made more than 15 days late."""
        if not self.payment_history:
            return 0.0
        chronic = sum(1 for r in self.payment_history if r.days_late > 15)
        return chronic / len(self.payment_history)


# ---------------------------------------------------------------------------
# Invoice models
# ---------------------------------------------------------------------------


class Invoice(BaseModel):
    """An outstanding (unpaid) invoice."""

    invoice_id: str
    customer_id: str
    amount: float = Field(gt=0)
    currency: str = Field(default="USD")
    issue_date: date
    due_date: date
    status: PaymentStatus = PaymentStatus.OVERDUE
    line_items: list[str] = Field(default_factory=list)
    notes: Optional[str] = None

    @computed_field  # type: ignore[misc]
    @property
    def days_outstanding(self) -> int:
        """Calendar days since the due date (negative = not yet due)."""
        return (date.today() - self.due_date).days

    @model_validator(mode="after")
    def due_after_issue(self) -> Invoice:
        if self.due_date < self.issue_date:
            raise ValueError("due_date must be on or after issue_date")
        return self


# ---------------------------------------------------------------------------
# Agent output models — Pydantic guardrails over LLM output
# ---------------------------------------------------------------------------


class DunningAction(BaseModel):
    """
    The structured output produced by the Collection Agent.

    This is the critical guardrail: the LLM CANNOT output free-form text.
    Every field is typed, validated, and audit-ready before it touches any
    downstream system.
    """

    customer_id: str
    invoice_id: str
    assigned_risk_tier: RiskTier
    recommended_tone: CommunicationTone
    recommended_channel: EscalationChannel
    email_subject: str = Field(min_length=5, max_length=120)
    email_draft_body: str = Field(min_length=50)
    next_escalation_days: int = Field(ge=1, le=90)
    justification: str = Field(
        description="Plain-English reasoning the agent used to reach this decision."
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ARHealthSnapshot(BaseModel):
    """
    Portfolio-level AR health summary (Pulse-style output).
    Gives a bird's-eye view of the entire receivables book.
    """

    snapshot_date: date = Field(default_factory=date.today)
    total_outstanding: float
    total_overdue: float
    weighted_avg_dso: float
    customers_at_risk: int
    high_critical_exposure: float  # $ sum of High + Critical tier invoices
    collection_probability_30d: float = Field(ge=0.0, le=1.0)
    top_risk_customer_ids: list[str]
    narrative: str  # LLM-generated plain English summary
