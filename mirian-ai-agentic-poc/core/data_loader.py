"""
core/data_loader.py
-------------------
Loads and validates mock data from JSON files into typed Pydantic models.

In a production system, these loaders would be replaced with ERP connectors
(NetSuite, QuickBooks, Xero, SAP) via the Mirian AI integration layer.
The interface contract — returning validated model instances — stays identical.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.models import Customer, Invoice

DATA_DIR = Path(__file__).parent.parent / "mock_data"


def load_customers() -> dict[str, Customer]:
    """
    Load all customers from customers.json.

    Returns a dict keyed by customer_id for O(1) lookup during agent runs.
    Raises ValueError if the JSON contains invalid data (Pydantic validation).
    """
    raw = json.loads((DATA_DIR / "customers.json").read_text())
    customers: dict[str, Customer] = {}
    for item in raw:
        customer = Customer.model_validate(item)
        customers[customer.customer_id] = customer
    return customers


def load_invoices() -> list[Invoice]:
    """
    Load all open/overdue invoices from unpaid_invoices.json.

    Returns a list sorted by days_outstanding descending so the most
    urgent invoices are processed first.
    """
    raw = json.loads((DATA_DIR / "unpaid_invoices.json").read_text())
    invoices = [Invoice.model_validate(item) for item in raw]
    # Most overdue first
    return sorted(invoices, key=lambda inv: inv.days_outstanding, reverse=True)
