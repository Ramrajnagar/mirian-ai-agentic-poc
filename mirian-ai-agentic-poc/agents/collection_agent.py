"""
agents/collection_agent.py
--------------------------
Behavioral AR Dunning Agent — the core agentic module.

Architecture:
  1. Risk tier is assigned DETERMINISTICALLY by core/finance_rules.py
  2. The LLM receives the tier, tone, and full customer context as a structured prompt
  3. The LLM's output is validated against DunningAction (Pydantic) — hallucinations
     in financial categories are structurally impossible
  4. The validated action is returned for human review before any email is sent

Supported backends: Anthropic Claude (default) | OpenAI GPT-4o-mini (fallback)
Set LLM_PROVIDER=openai in .env to switch.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from core.finance_rules import (
    assign_risk_tier,
    get_escalation_days,
    get_recommended_channel,
    get_recommended_tone,
)
from core.models import (
    CommunicationTone,
    Customer,
    DunningAction,
    EscalationChannel,
    Invoice,
    RiskTier,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tone guidance injected into the prompt — keeps the LLM on-policy
# ---------------------------------------------------------------------------

TONE_GUIDANCE: dict[CommunicationTone, str] = {
    CommunicationTone.CONCIERGE: (
        "The customer is historically a reliable payer. Treat this as a courtesy reminder. "
        "Use warm, professional language. Acknowledge their strong track record. "
        "Assume the missed payment is an oversight, not negligence."
    ),
    CommunicationTone.COLLABORATIVE: (
        "The customer pays a bit slowly but has always paid. Adopt a collaborative, "
        "solution-oriented tone. Offer a payment plan if appropriate. "
        "Be direct about the overdue amount but avoid any threatening language."
    ),
    CommunicationTone.FIRM: (
        "This customer has a documented pattern of late payments. Be firm, factual, "
        "and professional. Clearly state consequences if payment is not received within the "
        "stated window. Do NOT use aggressive language, but do NOT soften deadlines."
    ),
    CommunicationTone.LEGAL_WARNING: (
        "This customer is a chronic non-payer and this invoice is severely overdue. "
        "State clearly that the account has been escalated. Reference that the matter "
        "may be referred to collections or legal counsel if not resolved immediately. "
        "Keep language professional but unambiguous about the seriousness."
    ),
}


# ---------------------------------------------------------------------------
# LLM client helpers
# ---------------------------------------------------------------------------


def _build_prompt(
    customer: Customer,
    invoice: Invoice,
    risk_tier: RiskTier,
    tone: CommunicationTone,
    channel: EscalationChannel,
    escalation_days: int,
) -> str:
    """Build the structured prompt sent to the LLM."""
    tone_instruction = TONE_GUIDANCE[tone]

    return f"""You are MIRIAN, a specialist AR Collections Agent embedded in an AI-native
Financial Operating System. Your output must be precise, audit-ready, and professional.

=== CUSTOMER PROFILE ===
ID:               {customer.customer_id}
Company:          {customer.company_name}
Contact:          {customer.contact_name} <{customer.contact_email}>
Industry:         {customer.industry}
Historical avg days late: {customer.average_days_late:.1f} days
Chronic late rate:        {customer.chronic_late_rate:.0%} of invoices paid >15 days late
Total invoices on record: {customer.payment_count}

=== OUTSTANDING INVOICE ===
Invoice ID:       {invoice.invoice_id}
Amount Due:       {invoice.currency} {invoice.amount:,.2f}
Due Date:         {invoice.due_date}
Days Outstanding: {invoice.days_outstanding} days
Line Items:       {', '.join(invoice.line_items) if invoice.line_items else 'Not specified'}

=== AGENT DECISION (DETERMINISTIC — DO NOT OVERRIDE) ===
Risk Tier:        {risk_tier.value}
Tone:             {tone.value}
Channel:          {channel.value}
Next escalation:  {escalation_days} days from today

=== TONE INSTRUCTION ===
{tone_instruction}

=== YOUR TASK ===
Generate a collections communication for this customer. You must return a JSON object
with EXACTLY these fields and NO others:

{{
  "email_subject": "<concise, professional subject line — max 120 chars>",
  "email_draft_body": "<full email body — professional, signed off as 'MIRIAN Finance Operations'>",
  "justification": "<2-3 sentence plain-English explanation of why this tone and approach was chosen>"
}}

Rules:
- Do NOT invent financial figures not provided above
- Do NOT reference risk tier or internal scoring in the email body (customer-facing only)
- The email must include: invoice ID, amount, due date, and clear call to action
- Do NOT include any markdown formatting in the email body — plain text only
- Return ONLY the JSON object. No preamble, no explanation outside the JSON.
"""


def _call_anthropic(prompt: str) -> dict:
    """Call Anthropic Claude using the messages API."""
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "anthropic package not installed. Run: pip install anthropic"
        ) from exc

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in environment.")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = message.content[0].text.strip()
    return json.loads(raw_text)


def _call_openai(prompt: str) -> dict:
    """Call OpenAI GPT-4o-mini using structured output parsing."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = response.choices[0].message.content or "{}"
    return json.loads(raw_text)


def _call_llm(prompt: str) -> dict:
    """Route to the configured LLM backend."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        logger.debug("Using OpenAI backend.")
        return _call_openai(prompt)
    logger.debug("Using Anthropic backend.")
    return _call_anthropic(prompt)


# ---------------------------------------------------------------------------
# Public agent interface
# ---------------------------------------------------------------------------


def run_collection_agent(
    customer: Customer,
    invoice: Invoice,
) -> DunningAction:
    """
    Run the full dunning pipeline for one customer-invoice pair.

    Steps:
      1. Deterministic risk assessment (no LLM)
      2. Prompt construction with guardrails
      3. LLM call for language generation
      4. Pydantic validation of LLM output (rejects any malformed response)
      5. Return a fully typed, audit-ready DunningAction

    Raises:
      RuntimeError — if the LLM returns output that fails schema validation
      EnvironmentError — if the required API key is missing
    """
    logger.info(
        "Processing invoice %s for customer %s (%s)",
        invoice.invoice_id,
        customer.customer_id,
        customer.company_name,
    )

    # Step 1: Deterministic decision-making
    risk_tier = assign_risk_tier(customer, invoice)
    tone = get_recommended_tone(risk_tier)
    channel = get_recommended_channel(risk_tier)
    escalation_days = get_escalation_days(risk_tier)

    logger.debug(
        "Risk=%s | Tone=%s | Channel=%s | EscalationDays=%d",
        risk_tier,
        tone,
        channel,
        escalation_days,
    )

    # Step 2: Build prompt
    prompt = _build_prompt(
        customer, invoice, risk_tier, tone, channel, escalation_days
    )

    # Step 3: Call LLM for language generation only
    raw_output = _call_llm(prompt)

    # Step 4: Validate and assemble the final typed action
    action = DunningAction(
        customer_id=customer.customer_id,
        invoice_id=invoice.invoice_id,
        assigned_risk_tier=risk_tier,
        recommended_tone=tone,
        recommended_channel=channel,
        email_subject=raw_output["email_subject"],
        email_draft_body=raw_output["email_draft_body"],
        next_escalation_days=escalation_days,
        justification=raw_output["justification"],
    )

    logger.info(
        "Action generated: risk=%s | escalate_in=%dd",
        action.assigned_risk_tier,
        action.next_escalation_days,
    )
    return action


def batch_run(
    customers: dict[str, Customer],
    invoices: list[Invoice],
    limit: Optional[int] = None,
) -> list[DunningAction]:
    """
    Process a batch of invoices, skipping any where the customer is not found.

    Args:
        customers: Dict of customer_id → Customer (from data_loader)
        invoices:  List of Invoice objects sorted by urgency
        limit:     Max invoices to process (useful for demos / rate-limit management)

    Returns:
        List of DunningAction — one per successfully processed invoice
    """
    actions: list[DunningAction] = []
    invoices_to_process = invoices[:limit] if limit else invoices

    for invoice in invoices_to_process:
        customer = customers.get(invoice.customer_id)
        if not customer:
            logger.warning(
                "No customer record for invoice %s (customer_id=%s). Skipping.",
                invoice.invoice_id,
                invoice.customer_id,
            )
            continue

        try:
            action = run_collection_agent(customer, invoice)
            actions.append(action)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to process invoice %s: %s", invoice.invoice_id, exc
            )

    return actions
