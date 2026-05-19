"""
tests/test_agent.py
-------------------
Integration tests for the Collection Agent.

The LLM backend is mocked — these tests verify:
  1. The agent pipeline wires risk + tone + LLM output correctly
  2. Pydantic validation rejects malformed LLM output
  3. Batch processing skips missing customers gracefully
  4. Data loader produces correctly typed models from JSON

Run with: pytest tests/test_agent.py -v
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.models import Customer, DunningAction, Invoice, PaymentRecord, PaymentStatus, RiskTier


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _prompt_payer(customer_id: str = "CUST-001") -> Customer:
    return Customer(
        customer_id=customer_id,
        company_name="Prompt Payer Inc.",
        contact_name="Alice Good",
        contact_email="alice@good.com",
        industry="Technology",
        credit_limit=100_000,
        payment_history=[
            PaymentRecord(
                invoice_id=f"INV-{i:04d}",
                invoice_amount=10_000,
                due_date=date(2024, 1, i + 1),
                paid_date=date(2024, 1, i + 1),
                days_late=0,
            )
            for i in range(5)
        ],
    )


def _slow_payer(customer_id: str = "CUST-002") -> Customer:
    return Customer(
        customer_id=customer_id,
        company_name="Slow Payer LLC",
        contact_name="Bob Late",
        contact_email="bob@late.com",
        industry="Retail",
        credit_limit=200_000,
        payment_history=[
            PaymentRecord(
                invoice_id=f"INV-{i:04d}",
                invoice_amount=15_000,
                due_date=date(2024, 1, i + 1),
                paid_date=date(2024, 1, i + 1) + timedelta(days=45),
                days_late=45,
            )
            for i in range(5)
        ],
    )


def _overdue_invoice(customer_id: str = "CUST-001", days: int = 10) -> Invoice:
    due = date.today() - timedelta(days=days)
    return Invoice(
        invoice_id="INV-TEST-001",
        customer_id=customer_id,
        amount=25_000,
        currency="USD",
        issue_date=due - timedelta(days=30),
        due_date=due,
        status=PaymentStatus.OVERDUE,
        line_items=["Software License — $25,000"],
    )


def _mock_llm_output(subject: str = "Payment Reminder", body: str = None) -> dict:
    return {
        "email_subject": subject,
        "email_draft_body": body or (
            "Dear Alice Good,\n\n"
            "This is a reminder that invoice INV-TEST-001 for $25,000.00 "
            "was due on 2025-04-01. Please arrange payment at your earliest convenience.\n\n"
            "Best regards,\nMIRIAN Finance Operations"
        ),
        "justification": (
            "Customer has an excellent payment history with near-zero average delay. "
            "A gentle concierge tone is appropriate for what appears to be an oversight."
        ),
    }


# ---------------------------------------------------------------------------
# Collection agent tests
# ---------------------------------------------------------------------------


class TestCollectionAgent:
    @patch("agents.collection_agent._call_llm")
    def test_produces_valid_dunning_action_for_prompt_payer(self, mock_llm):
        """Agent should produce a Low-risk, Concierge-tone action for a prompt payer."""
        from agents.collection_agent import run_collection_agent

        mock_llm.return_value = _mock_llm_output()

        customer = _prompt_payer()
        invoice = _overdue_invoice(days=5)
        action = run_collection_agent(customer, invoice)

        assert isinstance(action, DunningAction)
        assert action.assigned_risk_tier == RiskTier.LOW
        assert action.customer_id == customer.customer_id
        assert action.invoice_id == invoice.invoice_id
        assert len(action.email_draft_body) > 50

    @patch("agents.collection_agent._call_llm")
    def test_produces_high_risk_action_for_chronic_slow_payer(self, mock_llm):
        """Agent should produce High/Critical risk action for chronic late payer."""
        from agents.collection_agent import run_collection_agent

        mock_llm.return_value = _mock_llm_output(
            subject="Urgent: Outstanding Payment Required — Invoice INV-TEST-001",
            body=(
                "Dear Bob Late,\n\n"
                "Invoice INV-TEST-001 for $25,000.00 is significantly overdue. "
                "Immediate payment is required to avoid further escalation.\n\n"
                "Regards,\nMIRIAN Finance Operations"
            ),
        )

        customer = _slow_payer()
        invoice = _overdue_invoice(customer_id="CUST-002", days=50)
        action = run_collection_agent(customer, invoice)

        assert action.assigned_risk_tier in (RiskTier.HIGH, RiskTier.CRITICAL)
        assert action.next_escalation_days <= 3

    @patch("agents.collection_agent._call_llm")
    def test_missing_llm_field_raises_validation_error(self, mock_llm):
        """If LLM omits a required field, Pydantic should raise a validation error."""
        from agents.collection_agent import run_collection_agent

        # LLM returns output missing 'email_subject'
        mock_llm.return_value = {
            "email_draft_body": "Some body text that is long enough to pass validation.",
            "justification": "Some justification text.",
            # 'email_subject' missing intentionally
        }

        customer = _prompt_payer()
        invoice = _overdue_invoice(days=5)

        with pytest.raises(Exception):  # Pydantic ValidationError or KeyError
            run_collection_agent(customer, invoice)

    @patch("agents.collection_agent._call_llm")
    def test_batch_skips_missing_customer(self, mock_llm):
        """Batch runner should skip invoices with no matching customer record."""
        from agents.collection_agent import batch_run

        mock_llm.return_value = _mock_llm_output()

        customers = {"CUST-001": _prompt_payer("CUST-001")}
        invoices = [
            _overdue_invoice(customer_id="CUST-001", days=5),
            _overdue_invoice(customer_id="CUST-UNKNOWN", days=10),  # no customer
        ]

        actions = batch_run(customers, invoices)
        assert len(actions) == 1
        assert actions[0].customer_id == "CUST-001"

    @patch("agents.collection_agent._call_llm")
    def test_batch_limit_respected(self, mock_llm):
        """Batch runner should respect the limit parameter."""
        from agents.collection_agent import batch_run

        mock_llm.return_value = _mock_llm_output()

        customers = {
            "CUST-001": _prompt_payer("CUST-001"),
            "CUST-002": _slow_payer("CUST-002"),
        }
        invoices = [
            _overdue_invoice(customer_id="CUST-001", days=5),
            _overdue_invoice(customer_id="CUST-002", days=50),
        ]

        actions = batch_run(customers, invoices, limit=1)
        assert len(actions) == 1

    @patch("agents.collection_agent._call_llm")
    def test_action_timestamps_populated(self, mock_llm):
        """generated_at should be set automatically."""
        from agents.collection_agent import run_collection_agent

        mock_llm.return_value = _mock_llm_output()
        action = run_collection_agent(_prompt_payer(), _overdue_invoice())
        assert action.generated_at is not None


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------


class TestDataLoader:
    def test_load_customers_returns_dict(self):
        from core.data_loader import load_customers

        customers = load_customers()
        assert isinstance(customers, dict)
        assert len(customers) > 0
        for cid, customer in customers.items():
            assert isinstance(customer, Customer)
            assert cid == customer.customer_id

    def test_load_invoices_returns_sorted_list(self):
        from core.data_loader import load_invoices

        invoices = load_invoices()
        assert isinstance(invoices, list)
        assert len(invoices) > 0

        # Should be sorted descending by days_outstanding (most urgent first)
        for i in range(len(invoices) - 1):
            assert invoices[i].days_outstanding >= invoices[i + 1].days_outstanding

    def test_all_invoices_have_valid_customer(self):
        from core.data_loader import load_customers, load_invoices

        customers = load_customers()
        invoices = load_invoices()

        for invoice in invoices:
            assert invoice.customer_id in customers, (
                f"Invoice {invoice.invoice_id} references unknown customer {invoice.customer_id}"
            )
