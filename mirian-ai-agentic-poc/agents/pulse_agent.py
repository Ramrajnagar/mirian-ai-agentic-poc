"""
agents/pulse_agent.py
---------------------
Portfolio-level AR Health Agent — inspired by Mirian Pulse.

This agent ingests the entire receivables book and produces:
  1. A structured ARHealthSnapshot (typed, auditable)
  2. A plain-English narrative for the CFO / Controller

The deterministic layer calculates all numbers.
The LLM layer writes the CFO-ready narrative.
"""

from __future__ import annotations

import json
import logging
import os

from core.finance_rules import (
    assign_risk_tier,
    calculate_portfolio_dso,
    estimate_collection_probability,
)
from core.models import ARHealthSnapshot, Customer, Invoice, RiskTier

logger = logging.getLogger(__name__)


def _build_pulse_prompt(snapshot: ARHealthSnapshot) -> str:
    return f"""You are MIRIAN Pulse, an AI financial analyst embedded in a CFO operating system.
You are writing a concise, executive-ready AR health narrative for a CFO morning briefing.

=== PORTFOLIO SNAPSHOT (AS OF {snapshot.snapshot_date}) ===
Total Outstanding AR:          ${snapshot.total_outstanding:,.0f}
Total Overdue (past due):      ${snapshot.total_overdue:,.0f}
Weighted Avg DSO:              {snapshot.weighted_avg_dso:.1f} days
Customers at risk (High/Critical): {snapshot.customers_at_risk}
High/Critical exposure:        ${snapshot.high_critical_exposure:,.0f}
Estimated 30-day collection probability: {snapshot.collection_probability_30d:.0%}
Top at-risk customer IDs:      {', '.join(snapshot.top_risk_customer_ids) or 'None'}

=== YOUR TASK ===
Write a 3-4 sentence plain-English AR health narrative for a CFO.
Be specific with numbers. Highlight the single biggest risk and one actionable recommendation.
Do NOT use bullet points. Write in flowing, executive prose.
Return ONLY the narrative text — no JSON, no preamble.
"""


def _call_llm(prompt: str) -> str:
    """Call the configured LLM backend and return raw text (not JSON)."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package not installed.") from exc
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set.")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    # Default: Anthropic
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package not installed.") from exc
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set.")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def run_pulse_agent(
    customers: dict[str, Customer],
    invoices: list[Invoice],
) -> ARHealthSnapshot:
    """
    Generate a portfolio-level AR health snapshot with LLM narrative.

    Steps:
      1. Calculate all metrics deterministically
      2. Identify high/critical risk customers
      3. LLM writes the CFO narrative grounded in the computed numbers
      4. Returns a fully typed ARHealthSnapshot
    """
    total_outstanding = sum(inv.amount for inv in invoices)
    total_overdue = sum(
        inv.amount for inv in invoices if inv.days_outstanding > 0
    )
    dso = calculate_portfolio_dso(list(customers.values()), invoices)

    # Score each invoice and identify at-risk ones
    risk_by_customer: dict[str, RiskTier] = {}
    collection_probs: list[float] = []
    high_critical_exposure = 0.0

    for inv in invoices:
        customer = customers.get(inv.customer_id)
        if not customer:
            continue
        tier = assign_risk_tier(customer, inv)
        risk_by_customer[inv.customer_id] = tier

        prob = estimate_collection_probability(customer, inv)
        collection_probs.append(prob)

        if tier in (RiskTier.HIGH, RiskTier.CRITICAL):
            high_critical_exposure += inv.amount

    at_risk_customer_ids = [
        cid
        for cid, tier in risk_by_customer.items()
        if tier in (RiskTier.HIGH, RiskTier.CRITICAL)
    ]
    avg_prob = (
        sum(collection_probs) / len(collection_probs) if collection_probs else 0.0
    )

    # Build partial snapshot for prompt (no narrative yet)
    snapshot = ARHealthSnapshot(
        total_outstanding=round(total_outstanding, 2),
        total_overdue=round(total_overdue, 2),
        weighted_avg_dso=dso,
        customers_at_risk=len(at_risk_customer_ids),
        high_critical_exposure=round(high_critical_exposure, 2),
        collection_probability_30d=round(avg_prob, 4),
        top_risk_customer_ids=at_risk_customer_ids[:5],
        narrative="",  # filled in next
    )

    # LLM writes the narrative — grounded only in the computed numbers above
    logger.info("Generating Pulse narrative via LLM...")
    prompt = _build_pulse_prompt(snapshot)
    narrative = _call_llm(prompt)

    # Return final snapshot with narrative attached
    return snapshot.model_copy(update={"narrative": narrative})
