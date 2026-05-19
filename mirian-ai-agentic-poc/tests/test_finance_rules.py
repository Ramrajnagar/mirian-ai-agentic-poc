"""
tests/test_finance_rules.py
----------------------------
Unit tests for the deterministic finance rules engine.

These tests cover:
  - Risk tier assignment across all scenarios
  - Tone and channel mapping
  - DSO calculation
  - Collection probability bounds
  - Edge cases (no history, maxed credit limit, very overdue)

All tests are pure — no LLM calls, no I/O, no network.
Run with: pytest tests/test_finance_rules.py -v
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from core.finance_rules import (
    assign_risk_tier,
    calculate_portfolio_dso,
    estimate_collection_probability,
    get_escalation_days,
    get_recommended_channel,
    get_recommended_tone,
)
from core.models import (
    CommunicationTone,
    Customer,
    EscalationChannel,
    Invoice,
    PaymentRecord,
    PaymentStatus,
    RiskTier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_customer(
    customer_id: str = "CUST-TEST",
    avg_days_late_per_invoice: float = 0,
    num_invoices: int = 5,
    credit_limit: float = 100_000,
) -> Customer:
    """Helper to create a Customer with synthetic payment history."""
    history = [
        PaymentRecord(
            invoice_id=f"INV-{i:04d}",
            invoice_amount=10_000,
            due_date=date(2024, 1, i + 1),
            paid_date=date(2024, 1, i + 1) + timedelta(days=int(avg_days_late_per_invoice)),
            days_late=int(avg_days_late_per_invoice),
        )
        for i in range(num_invoices)
    ]
    return Customer(
        customer_id=customer_id,
        company_name=f"Test Corp {customer_id}",
        contact_name="Test Contact",
        contact_email=f"test@{customer_id.lower()}.com",
        industry="Technology",
        credit_limit=credit_limit,
        payment_history=history,
    )


def _make_invoice(
    days_overdue: int = 10,
    amount: float = 10_000,
    customer_id: str = "CUST-TEST",
) -> Invoice:
    """Helper to create an Invoice that is `days_overdue` days past due."""
    due = date.today() - timedelta(days=days_overdue)
    return Invoice(
        invoice_id="INV-TEST-001",
        customer_id=customer_id,
        amount=amount,
        currency="USD",
        issue_date=due - timedelta(days=30),
        due_date=due,
        status=PaymentStatus.OVERDUE,
    )


# ---------------------------------------------------------------------------
# Risk tier assignment
# ---------------------------------------------------------------------------


class TestAssignRiskTier:
    def test_low_risk_prompt_payer(self):
        customer = _make_customer(avg_days_late_per_invoice=5)
        invoice = _make_invoice(days_overdue=5)
        assert assign_risk_tier(customer, invoice) == RiskTier.LOW

    def test_medium_risk_moderate_history(self):
        customer = _make_customer(avg_days_late_per_invoice=20)
        invoice = _make_invoice(days_overdue=20)
        assert assign_risk_tier(customer, invoice) == RiskTier.MEDIUM

    def test_high_risk_chronic_slow_payer(self):
        customer = _make_customer(avg_days_late_per_invoice=45)
        invoice = _make_invoice(days_overdue=45)
        assert assign_risk_tier(customer, invoice) == RiskTier.HIGH

    def test_critical_risk_very_overdue(self):
        customer = _make_customer(avg_days_late_per_invoice=5)
        invoice = _make_invoice(days_overdue=95)  # >90 days triggers Critical override
        assert assign_risk_tier(customer, invoice) == RiskTier.CRITICAL

    def test_critical_chronic_nonpayer(self):
        """A customer who pays >15 days late >50% of the time AND invoice is >45 days out."""
        history = [
            PaymentRecord(
                invoice_id=f"INV-{i:04d}",
                invoice_amount=10_000,
                due_date=date(2024, 1, 1),
                paid_date=date(2024, 1, 1) + timedelta(days=60),
                days_late=60,  # all >15 days → chronic_late_rate = 1.0
            )
            for i in range(5)
        ]
        customer = Customer(
            customer_id="CUST-CHRONIC",
            company_name="Chronic Inc.",
            contact_name="Never Pays",
            contact_email="bad@example.com",
            industry="Unknown",
            credit_limit=100_000,
            payment_history=history,
        )
        invoice = _make_invoice(days_overdue=50)
        assert assign_risk_tier(customer, invoice) == RiskTier.CRITICAL

    def test_low_risk_bumped_to_medium_if_overdue_30_days(self):
        """Even a great payer bumps to Medium if current invoice is >30 days overdue."""
        customer = _make_customer(avg_days_late_per_invoice=5)
        invoice = _make_invoice(days_overdue=35)
        assert assign_risk_tier(customer, invoice) == RiskTier.MEDIUM

    def test_no_payment_history(self):
        """Customer with no history should not crash — defaults based on days_outstanding."""
        customer = Customer(
            customer_id="CUST-NEW",
            company_name="New Co",
            contact_name="John New",
            contact_email="john@new.com",
            industry="Retail",
            credit_limit=50_000,
            payment_history=[],
        )
        invoice = _make_invoice(days_overdue=10)
        tier = assign_risk_tier(customer, invoice)
        assert tier in RiskTier  # should not raise


# ---------------------------------------------------------------------------
# Tone and channel mapping
# ---------------------------------------------------------------------------


class TestToneAndChannelMapping:
    @pytest.mark.parametrize(
        "tier,expected_tone",
        [
            (RiskTier.LOW, CommunicationTone.CONCIERGE),
            (RiskTier.MEDIUM, CommunicationTone.COLLABORATIVE),
            (RiskTier.HIGH, CommunicationTone.FIRM),
            (RiskTier.CRITICAL, CommunicationTone.LEGAL_WARNING),
        ],
    )
    def test_tone_mapping(self, tier, expected_tone):
        assert get_recommended_tone(tier) == expected_tone

    @pytest.mark.parametrize(
        "tier,expected_channel",
        [
            (RiskTier.LOW, EscalationChannel.EMAIL),
            (RiskTier.MEDIUM, EscalationChannel.EMAIL),
            (RiskTier.HIGH, EscalationChannel.PHONE),
            (RiskTier.CRITICAL, EscalationChannel.LEGAL),
        ],
    )
    def test_channel_mapping(self, tier, expected_channel):
        assert get_recommended_channel(tier) == expected_channel

    @pytest.mark.parametrize(
        "tier,max_days",
        [
            (RiskTier.LOW, 14),
            (RiskTier.MEDIUM, 7),
            (RiskTier.HIGH, 3),
            (RiskTier.CRITICAL, 1),
        ],
    )
    def test_escalation_days(self, tier, max_days):
        assert get_escalation_days(tier) == max_days


# ---------------------------------------------------------------------------
# Collection probability
# ---------------------------------------------------------------------------


class TestCollectionProbability:
    def test_probability_in_range(self):
        customer = _make_customer(avg_days_late_per_invoice=10)
        invoice = _make_invoice(days_overdue=15)
        prob = estimate_collection_probability(customer, invoice)
        assert 0.0 <= prob <= 1.0

    def test_prompt_payer_high_probability(self):
        customer = _make_customer(avg_days_late_per_invoice=2)
        invoice = _make_invoice(days_overdue=5)
        prob = estimate_collection_probability(customer, invoice)
        assert prob > 0.5, f"Expected > 0.5, got {prob}"

    def test_chronic_payer_lower_probability(self):
        customer = _make_customer(avg_days_late_per_invoice=60)
        invoice = _make_invoice(days_overdue=80)
        prob = estimate_collection_probability(customer, invoice)
        assert prob < 0.5, f"Expected < 0.5, got {prob}"

    def test_oversized_invoice_reduces_probability(self):
        """Invoice at 95% of credit limit should reduce collection probability."""
        credit_limit = 100_000
        customer = _make_customer(avg_days_late_per_invoice=5, credit_limit=credit_limit)
        small_invoice = _make_invoice(days_overdue=5, amount=5_000)
        large_invoice = _make_invoice(days_overdue=5, amount=95_000)

        prob_small = estimate_collection_probability(customer, small_invoice)
        prob_large = estimate_collection_probability(customer, large_invoice)

        assert prob_small > prob_large


# ---------------------------------------------------------------------------
# Portfolio DSO
# ---------------------------------------------------------------------------


class TestPortfolioDSO:
    def test_dso_empty_portfolio(self):
        assert calculate_portfolio_dso([], []) == 0.0

    def test_dso_positive(self):
        customer = _make_customer()
        invoices = [_make_invoice(days_overdue=30, amount=50_000)]
        dso = calculate_portfolio_dso([customer], invoices)
        assert dso == 30.0

    def test_dso_weighted(self):
        """DSO should weight by invoice amount."""
        customer = _make_customer()
        invoices = [
            _make_invoice(days_overdue=10, amount=90_000),  # large, short overdue
            _make_invoice(days_overdue=90, amount=10_000),  # small, long overdue
        ]
        dso = calculate_portfolio_dso([customer], invoices)
        # Weighted: (90000*10 + 10000*90) / 100000 = 18.0
        assert abs(dso - 18.0) < 0.01


# ---------------------------------------------------------------------------
# Model validation edge cases
# ---------------------------------------------------------------------------


class TestModelValidation:
    def test_invoice_due_before_issue_raises(self):
        with pytest.raises(ValueError, match="due_date must be on or after issue_date"):
            Invoice(
                invoice_id="INV-BAD",
                customer_id="CUST-001",
                amount=1000,
                issue_date=date(2025, 6, 1),
                due_date=date(2025, 5, 1),  # before issue date
                status=PaymentStatus.OVERDUE,
            )

    def test_customer_average_days_late_no_history(self):
        customer = Customer(
            customer_id="CUST-EMPTY",
            company_name="Empty History Co",
            contact_name="Nobody",
            contact_email="nobody@empty.com",
            industry="Other",
            credit_limit=10_000,
            payment_history=[],
        )
        assert customer.average_days_late == 0.0
        assert customer.chronic_late_rate == 0.0

    def test_invoice_amount_must_be_positive(self):
        with pytest.raises(ValueError):
            Invoice(
                invoice_id="INV-ZERO",
                customer_id="CUST-001",
                amount=-500,
                issue_date=date(2025, 1, 1),
                due_date=date(2025, 2, 1),
                status=PaymentStatus.OVERDUE,
            )
