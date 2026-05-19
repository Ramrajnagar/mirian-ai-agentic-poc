"""
main.py
-------
CLI entry point for the Mirian AI Agentic Finance PoC.

Usage:
  python main.py                        # Run full demo (Pulse + all invoices)
  python main.py --mode pulse           # AR portfolio health snapshot only
  python main.py --mode dunning         # Dunning agent for all open invoices
  python main.py --mode dunning --limit 2  # Process only 2 invoices
  python main.py --output results.json  # Save actions to JSON file
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any other imports that check env vars
load_dotenv()

from agents.collection_agent import batch_run
from agents.pulse_agent import run_pulse_agent
from core.data_loader import load_customers, load_invoices
from core.models import DunningAction

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mirian-poc")


# ---------------------------------------------------------------------------
# Terminal colour helpers (no external deps)
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
DIM = "\033[2m"

TIER_COLORS = {
    "Low": GREEN,
    "Medium": YELLOW,
    "High": RED,
    "Critical": MAGENTA,
}


def cprint(text: str, color: str = RESET, bold: bool = False) -> None:
    prefix = BOLD if bold else ""
    print(f"{prefix}{color}{text}{RESET}")


def divider(char: str = "─", width: int = 72) -> None:
    cprint(char * width, DIM)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_banner() -> None:
    cprint("\n" + "═" * 72, CYAN)
    cprint(
        "  MIRIAN AI  │  Agentic AR Dunning PoC  │  "
        f"Snapshot: {date.today()}",
        CYAN,
        bold=True,
    )
    cprint("═" * 72 + "\n", CYAN)


def print_pulse(snapshot) -> None:
    cprint("▶  MIRIAN PULSE — Portfolio AR Health", CYAN, bold=True)
    divider()
    print(f"  Total Outstanding AR :  {BOLD}${snapshot.total_outstanding:>12,.2f}{RESET}")
    print(f"  Total Overdue        :  {RED}${snapshot.total_overdue:>12,.2f}{RESET}")
    print(f"  Weighted Avg DSO     :  {snapshot.weighted_avg_dso:.1f} days")
    print(
        f"  Customers at Risk    :  "
        f"{RED if snapshot.customers_at_risk > 0 else GREEN}"
        f"{snapshot.customers_at_risk}{RESET}"
    )
    print(f"  High/Critical Exposure: ${snapshot.high_critical_exposure:>10,.2f}")
    print(
        f"  30-Day Collection Prob: "
        f"{snapshot.collection_probability_30d:.0%}"
    )
    divider()
    cprint("\n  CFO Narrative:", BOLD)
    print(f"\n  {snapshot.narrative}\n")
    divider()


def print_action(idx: int, action: DunningAction) -> None:
    tier_color = TIER_COLORS.get(action.assigned_risk_tier.value, RESET)
    cprint(
        f"\n  [{idx}]  Invoice {action.invoice_id}  │  Customer {action.customer_id}",
        BOLD,
    )
    print(
        f"       Risk Tier    : {tier_color}{action.assigned_risk_tier.value}{RESET}  │  "
        f"Tone: {action.recommended_tone.value}  │  "
        f"Channel: {action.recommended_channel.value}"
    )
    print(f"       Escalate in : {action.next_escalation_days} days")
    print(f"       Justification: {DIM}{action.justification}{RESET}")
    divider("· ", 36)
    cprint(f"  Subject: {action.email_subject}", YELLOW)
    print()
    # Indent the email body for readability
    for line in action.email_draft_body.splitlines():
        print(f"  {line}")
    divider()


def save_actions(actions: list[DunningAction], path: str) -> None:
    output = [
        json.loads(action.model_dump_json())
        for action in actions
    ]
    Path(path).write_text(json.dumps(output, indent=2, default=str))
    cprint(f"\n  ✓  Results saved to {path}", GREEN, bold=True)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_demo(mode: str, limit: int | None, output_path: str | None) -> None:
    print_banner()

    logger.info("Loading customer and invoice data...")
    customers = load_customers()
    invoices = load_invoices()
    logger.info(
        "Loaded %d customers and %d open invoices.",
        len(customers),
        len(invoices),
    )

    # --- Pulse mode ---
    if mode in ("pulse", "full"):
        cprint("\n  Running Pulse Agent...\n", DIM)
        snapshot = run_pulse_agent(customers, invoices)
        print_pulse(snapshot)

    # --- Dunning mode ---
    if mode in ("dunning", "full"):
        cprint("\n▶  COLLECTION AGENT — Dunning Workflow\n", CYAN, bold=True)
        actions = batch_run(customers, invoices, limit=limit)

        if not actions:
            cprint("  No actions generated. Check your data or API key.", YELLOW)
        else:
            for i, action in enumerate(actions, start=1):
                print_action(i, action)

            cprint(
                f"\n  ✓  {len(actions)} dunning action(s) generated successfully.",
                GREEN,
                bold=True,
            )

            if output_path:
                save_actions(actions, output_path)

    cprint("\n" + "═" * 72 + "\n", CYAN)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mirian AI — Agentic AR Dunning PoC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "pulse", "dunning"],
        default="full",
        help="Which agents to run (default: full — runs both Pulse and Dunning)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of invoices to process in dunning mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save dunning actions as JSON (e.g. results.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate at least one API key is set before doing any work
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    key_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.getenv(key_name):
        cprint(
            f"\n  ✗  {key_name} not found in environment.\n"
            f"     Copy .env.example to .env and add your key.\n",
            RED,
            bold=True,
        )
        sys.exit(1)

    try:
        run_demo(mode=args.mode, limit=args.limit, output_path=args.output)
    except KeyboardInterrupt:
        cprint("\n  Interrupted by user.", YELLOW)
        sys.exit(0)


if __name__ == "__main__":
    main()
