"""
core/finance_rules.py
---------------------
Deterministic financial logic — pure math, no LLM involved.

This layer handles everything that MUST be correct and reproducible:
  - DSO calculation
  - Risk tier assignment
  - Recommended tone and escalation channel
  - Collection probability estimation

Design principle: LLMs draft language; rules engines make decisions.
Every function here is stateless and unit-testable.
"""

from __future__ import annotations

import math

from core.models import (
    CommunicationTone,
    Customer,
    EscalationChannel,
    Invoice,
    RiskTier,
)


# ---------------------------------------------------------------------------
# Constants — tune these for each client's credit policy
# ---------------------------------------------------------------------------

DSO_THRESHOLDS = {
    RiskTier.LOW: (0, 15),       # avg days late  0–15  → Low
    RiskTier.MEDIUM: (15, 30),   #                15–30 → Medium
    RiskTier.HIGH: (30, 60),     #                30–60 → High
    RiskTier.CRITICAL: (60, math.inf),  #         60+   → Critical
}

TONE_MAP: dict[RiskTier, CommunicationTone] = {
    RiskTier.LOW: CommunicationTone.CONCIERGE,
    RiskTier.MEDIUM: CommunicationTone.COLLABORATIVE,
    RiskTier.HIGH: CommunicationTone.FIRM,
    RiskTier.CRITICAL: CommunicationTone.LEGAL_WARNING,
}

CHANNEL_MAP: dict[RiskTier, EscalationChannel] = {
    RiskTier.LOW: EscalationChannel.EMAIL,
    RiskTier.MEDIUM: EscalationChannel.EMAIL,
    RiskTier.HIGH: EscalationChannel.PHONE,
    RiskTier.CRITICAL: EscalationChannel.LEGAL,
}

ESCALATION_DAYS_MAP: dict[RiskTier, int] = {
    RiskTier.LOW: 14,
    RiskTier.MEDIUM: 7,
    RiskTier.HIGH: 3,
    RiskTier.CRITICAL: 1,
}


# ---------------------------------------------------------------------------
# Risk Engine
# ---------------------------------------------------------------------------


def assign_risk_tier(customer: Customer, invoice: Invoice) -> RiskTier:
    """
    Assign a risk tier based on behavioral and situational signals.

    Signals used (in priority order):
      1. Current invoice days outstanding (situational urgency)
      2. Customer's historical average days late (behavioral pattern)
      3. Customer's chronic late rate (frequency of bad behaviour)

    Returns one of: Low | Medium | High | Critical
    """
    avg_late = customer.average_days_late
    chronic_rate = customer.chronic_late_rate
    days_out = invoice.days_outstanding

    # Override to Critical if very overdue regardless of history
    if days_out > 90:
        return RiskTier.CRITICAL

    # Override to High/Critical if chronic non-payer
    if chronic_rate >= 0.5 and avg_late >= 30:
        return RiskTier.CRITICAL if days_out > 45 else RiskTier.HIGH

    # Standard DSO-based classification
    for tier, (low, high) in DSO_THRESHOLDS.items():
        if low <= avg_late < high:
            # Bump up one tier if current invoice is significantly overdue
            if days_out > 30 and tier == RiskTier.LOW:
                return RiskTier.MEDIUM
            if days_out > 60 and tier == RiskTier.MEDIUM:
                return RiskTier.HIGH
            return tier

    return RiskTier.CRITICAL


def get_recommended_tone(risk_tier: RiskTier) -> CommunicationTone:
    """Map risk tier → communication tone."""
    return TONE_MAP[risk_tier]


def get_recommended_channel(risk_tier: RiskTier) -> EscalationChannel:
    """Map risk tier → preferred escalation channel."""
    return CHANNEL_MAP[risk_tier]


def get_escalation_days(risk_tier: RiskTier) -> int:
    """Return days to wait before the next human escalation."""
    return ESCALATION_DAYS_MAP[risk_tier]


def estimate_collection_probability(customer: Customer, invoice: Invoice) -> float:
    """
    Estimate the probability of collecting within 30 days.

    Uses a simple logistic-style model over:
      - Historical payment rate (proxy: 1 - chronic_late_rate)
      - Days outstanding (recency decay)
      - Invoice size relative to credit limit (capacity signal)

    Returns a float in [0.0, 1.0].
    Note: In production, replace this with a trained ML model.
    """
    base = 1.0 - customer.chronic_late_rate  # baseline from history

    # Recency decay: each 30 days overdue reduces probability by ~20%
    recency_factor = max(0.0, 1.0 - (invoice.days_outstanding / 150))

    # Exposure factor: invoices > 80% of credit limit are riskier
    exposure_ratio = invoice.amount / customer.credit_limit
    exposure_factor = 1.0 if exposure_ratio < 0.8 else max(0.3, 1.0 - exposure_ratio)

    prob = base * recency_factor * exposure_factor
    return round(min(max(prob, 0.0), 1.0), 4)


def calculate_portfolio_dso(
    customers: list[Customer],
    invoices: list[Invoice],
) -> float:
    """
    Calculate weighted-average Days Sales Outstanding across the portfolio.

    DSO = (Outstanding AR / Total Credit Sales) * Days in Period

    Here we use a simplified 90-day period proxy suitable for a PoC.
    """
    if not invoices:
        return 0.0

    total_outstanding = sum(inv.amount for inv in invoices)
    if total_outstanding == 0:
        return 0.0

    weighted_days = sum(inv.amount * inv.days_outstanding for inv in invoices)
    return round(weighted_days / total_outstanding, 2)
