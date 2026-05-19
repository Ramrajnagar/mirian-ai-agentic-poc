# Agentic AR Dunning Agent  Mirian AI PoC

> A working proof-of-concept demonstrating an **AI-native Accounts Receivable automation agent**  built independently to show technical and product alignment with [Mirian AI](https://mirianai.com/).

[![CI](https://github.com/YOUR_USERNAME/mirian-ai-agentic-poc/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/mirian-ai-agentic-poc/actions)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Pydantic](https://img.shields.io/badge/pydantic-v2-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## The Problem

Finance teams lose weeks chasing the same invoices with the same generic emails. Every late customer gets the same message  regardless of whether they're a 10-year prompt payer who simply forgot, or a chronic non-payer who needs a legal escalation. This wastes AR team time, damages customer relationships, and leaves cash on the table.

## The Solution

This PoC implements two interconnected agents that mirror Mirian AI's core AR automation pillars:

| Agent | Mirrors Mirian Feature |
|---|---|
| **Behavioral Dunning Agent** | AI Collection Agent + Behavioral Customer Segmentation |
| **Pulse Agent** | Mirian Pulse + Risk Scoring Engine |

Both agents share a strict architectural principle: **deterministic logic makes decisions, LLMs generate language**. The model cannot override a risk tier, invent a figure, or produce output that breaks the financial pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     main.py (CLI)                        │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────────┐     ┌───────────────────────┐
│  pulse_agent.py     │     │  collection_agent.py  │
│  (Portfolio Health) │     │  (Per-Invoice Action) │
└──────────┬──────────┘     └──────────┬────────────┘
           │                           │
           └─────────────┬─────────────┘
                         ▼
            ┌────────────────────────┐
            │   core/finance_rules   │  ← Deterministic math
            │   core/models.py       │  ← Pydantic guardrails
            └────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │   LLM Backend          │
            │   Anthropic Claude     │  ← Language generation only
            │   (or OpenAI fallback) │
            └────────────────────────┘
```

### Key Design Decisions

**1. Deterministic risk before LLM.** The `assign_risk_tier()` function computes risk using historical average days late, chronic late rate, and current invoice age — before a single token is generated. The LLM receives the tier as a constraint, not a suggestion.

**2. Pydantic as a financial firewall.** Every LLM output is validated against `DunningAction` or `ARHealthSnapshot`. If the model hallucinates a risk tier or omits a required field, the pipeline raises a `ValidationError` before any email is drafted or sent.

**3. Dual-backend support.** Set `LLM_PROVIDER=openai` in `.env` to switch from Anthropic Claude to GPT-4o-mini without changing a line of business logic.

---

## Repository Structure

```
mirian-ai-agentic-poc/
├── .github/workflows/ci.yml      # CI: lint, type-check, test with coverage
├── agents/
│   ├── collection_agent.py       # Behavioral dunning — one invoice at a time
│   └── pulse_agent.py            # Portfolio AR health + CFO narrative
├── core/
│   ├── models.py                 # All Pydantic data models (single source of truth)
│   ├── finance_rules.py          # Deterministic risk engine — pure math, no LLM
│   └── data_loader.py            # ERP adapter (mock JSON → typed models)
├── mock_data/
│   ├── customers.json            # 5 customers across 4 risk profiles
│   └── unpaid_invoices.json      # 5 open invoices with realistic line items
├── tests/
│   ├── test_finance_rules.py     # 15+ unit tests for the risk engine
│   └── test_agent.py             # Integration tests with mocked LLM calls
├── .env.example                  # API key template
├── main.py                       # CLI entrypoint
├── pyproject.toml                # Ruff + mypy + pytest config
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- An Anthropic or OpenAI API key

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/mirian-ai-agentic-poc.git
cd mirian-ai-agentic-poc
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY (or OPENAI_API_KEY + LLM_PROVIDER=openai)
```

### 3. Run the full demo

```bash
python main.py
```

### 4. Run specific agents

```bash
python main.py --mode pulse           # Portfolio health snapshot only
python main.py --mode dunning         # Dunning agent for all invoices
python main.py --mode dunning --limit 2  # Process only 2 invoices
python main.py --output results.json  # Save actions to a JSON file
```

### 5. Run tests

```bash
pytest                          # All tests
pytest tests/test_finance_rules.py -v   # Unit tests only (no API key needed)
pytest --cov=. --cov-report=html        # With HTML coverage report
```

---

## The Risk Engine

Customer risk is scored deterministically across four tiers before any LLM call:

| Tier | Avg Days Late | Invoice Age Override | Tone | Channel | Escalate In |
|---|---|---|---|---|---|
| **Low** | 0–15 days | — | Concierge | Email | 14 days |
| **Medium** | 15–30 days | >30 days overdue | Collaborative | Email | 7 days |
| **High** | 30–60 days | >60 days overdue | Firm | Phone | 3 days |
| **Critical** | 60+ days | >90 days overdue | Legal Warning | Legal | 1 day |

The tier also incorporates a **chronic late rate** — the fraction of historical invoices paid more than 15 days late. A customer with a 60%+ chronic rate gets bumped to Critical regardless of average DSO.

---

## Pydantic Guardrails in Action

The `DunningAction` model is the contract between the LLM and the rest of the system. The LLM cannot produce anything that doesn't satisfy it:

```python
class DunningAction(BaseModel):
    customer_id: str
    invoice_id: str
    assigned_risk_tier: RiskTier          # Enum — only Low/Medium/High/Critical
    recommended_tone: CommunicationTone   # Enum — 4 valid values
    recommended_channel: EscalationChannel
    email_subject: str = Field(min_length=5, max_length=120)
    email_draft_body: str = Field(min_length=50)
    next_escalation_days: int = Field(ge=1, le=90)
    justification: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
```

In production, you'd extend this with an `approved_by` field and a status workflow before any email is dispatched.

---

## Sample Output

```
══════════════════════════════════════════════════════════════════════════
  MIRIAN AI  │  Agentic AR Dunning PoC  │  Snapshot: 2025-05-19
══════════════════════════════════════════════════════════════════════════

▶  MIRIAN PULSE — Portfolio AR Health
────────────────────────────────────────────────────────────────────────
  Total Outstanding AR :     $306,500.00
  Total Overdue        :     $306,500.00
  Weighted Avg DSO     :  52.3 days
  Customers at Risk    :  2
  High/Critical Exposure: $118,800.00
  30-Day Collection Prob: 61%

  CFO Narrative:
  Your AR portfolio carries $306,500 in outstanding receivables, all currently
  overdue with a weighted DSO of 52 days — elevated versus a healthy 30-day
  target. Two customers (Nexus Logistics and Brightstone Retail) account for
  $118,800 of high/critical exposure and require immediate escalation...

▶  COLLECTION AGENT — Dunning Workflow

  [1]  Invoice INV-2503-NEXUS  │  Customer CUST-003
       Risk Tier    : Critical  │  Tone: LegalWarning  │  Channel: legal
       Escalate in : 1 days
       Justification: Nexus Logistics has a documented pattern of 75-101 day
       delays across all historical invoices, with two invoices currently
       unpaid. This invoice at $91,000 is 77 days overdue...
```

---

## Production Roadmap

What it would take to move this from PoC to production-grade:

- [ ] **ERP Adapters** — Replace `data_loader.py` with real NetSuite / QuickBooks connectors (Mirian already has these)
- [ ] **Human-in-the-loop** — Add an `approved_by` field and webhook/email approval workflow before dispatch
- [ ] **Feedback loop** — Log which email tone actually resulted in payment; retrain tone model
- [ ] **Multi-language** — Extend `_build_prompt()` to accept a `locale` parameter for international AR
- [ ] **Audit log** — Persist every `DunningAction` to a database with full lineage
- [ ] **Real ML model** — Replace `estimate_collection_probability()` with a trained gradient boosting model on historical payment data

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.11+ | Type hints, `match` statements, speed |
| Data Validation | Pydantic v2 | Financial data integrity, LLM guardrails |
| LLM (primary) | Anthropic Claude | Best instruction-following for structured finance tasks |
| LLM (fallback) | OpenAI GPT-4o-mini | Cost-effective alternative |
| Testing | PyTest + pytest-cov | 80%+ coverage enforced in CI |
| Linting | Ruff | Fast, comprehensive |
| Type Checking | mypy | Catches bugs before runtime |
| CI | GitHub Actions | Matrix testing on Python 3.11 + 3.12 |

---

## About This Project

Built independently to demonstrate product alignment and technical execution for **Mirian AI** — the AI-native Financial Operating System for mid-market finance teams.

The architecture deliberately mirrors Mirian's stated product principles:
- *"Deterministic logic handles the math. Agentic reasoning handles the exceptions."*
- *"Every AI action is tracked and compliance-ready."*
- *"Agents that observe, reason, and act."*

---

*Questions or feedback? Open an issue or reach out directly.*
