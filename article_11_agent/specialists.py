"""The three domain specialists: Order Management, Returns and Refunds, Billing and Payments.

Every specialist has the same shape:
  1. Read the resolved order ID from state. Identifiers are never re-extracted here.
  2. Fetch facts from tools.py and compute analysis in code.
  3. Compute escalation triggers in code from those facts and STORE_POLICIES.
  4. Ask the LLM to assess the facts. It never sees triggers and never decides
     escalation.
  5. Return one result, stamped with the order ID it was fetched for.
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta

from pydantic import BaseModel

import llm
from prompts import BILLING_SYSTEM, ORDER_SYSTEM, RETURNS_SYSTEM, specialist_user
from routing import RESULT_FIELD
from state import ChatState, EscalationTrigger, SpecialistResult
from tools import (
    find_transactions_by_order,
    get_policy,
    lookup_order_by_id,
    lookup_return_by_order,
)

# All mock data is dated relative to this day. In production: datetime.now().
REFERENCE_DATE = datetime(2024, 2, 1)


def _days_since(date_str: str) -> int:
    return (REFERENCE_DATE - datetime.strptime(date_str, "%Y-%m-%d")).days


# ---------------------------------------------------------------------------
# Shared: the LLM assessment and the stamped result
# ---------------------------------------------------------------------------

class SpecialistAssessment(BaseModel):
    assessment: str
    recommended_action: str
    needs_judgment_review: bool = False
    review_reason: str = ""


def _finish(
    state: ChatState,
    name: str,
    order_id: str,
    facts: dict,
    triggers: list[EscalationTrigger],
    system: str,
) -> dict:
    """Run the LLM assessment and package the stamped result.

    The LLM sees the same facts dict the payload will carry, serialized as JSON,
    so what the model assessed and what the customer is told come from one source.
    """
    user = specialist_user(
        conversation=llm.transcript(state.get("messages", [])),
        clarification_facts=state.get("clarification_facts", {}),
        facts=json.dumps(facts, indent=2, default=str),
    )
    triggers = list(triggers)
    try:
        out = llm.structured(system, user, SpecialistAssessment, label=name)
        assessment, action = out.assessment, out.recommended_action
        if out.needs_judgment_review:
            triggers.append("specialist_judgment")
            print(f"[{name}] judgment review requested: {out.review_reason}")
    except ValueError as e:
        # A failed assessment goes to a human rather than to the customer.
        print(f"[{name}] assessment failed: {e}")
        assessment = "Assessment unavailable due to a processing error."
        action = "Manual review required."
        triggers.append("specialist_error")

    result: SpecialistResult = {
        "fetched_for": order_id,
        "facts": facts,
        "assessment": assessment,
        "recommended_action": action,
        "triggers": triggers,
    }
    print(f"[{name}] {order_id} | triggers: {triggers or 'none'}")
    return {RESULT_FIELD[name]: result}


# ---------------------------------------------------------------------------
# 4. Order Management
# ---------------------------------------------------------------------------

def order_management(state: ChatState) -> dict:
    order_id = state["resolution"]["order_id"]
    order = lookup_order_by_id(order_id)
    facts: dict = {"order": order}
    triggers: list[EscalationTrigger] = []

    if order["status"] == "delivered_unconfirmed":
        triggers.append("delivered_unconfirmed")

    if not order.get("actual_delivery") and order.get("expected_delivery"):
        overdue = _days_since(order["expected_delivery"])
        # Only a late order carries lateness facts. An order not yet due gets
        # none, so the reply has nothing to say about it being "not overdue".
        if overdue > 0:
            facts["days_overdue"] = overdue
            # A threshold is a rule, so code applies it and the LLM reads the result.
            facts["carrier_investigation_due"] = (
                overdue > get_policy("carrier_investigation_days")
            )
            if overdue > get_policy("overdue_escalation_days"):
                triggers.append("overdue_14_days")

    return _finish(state, "order_management", order_id, facts, triggers, ORDER_SYSTEM)


# ---------------------------------------------------------------------------
# 5. Returns and Refunds
# ---------------------------------------------------------------------------

def check_return_eligibility(order: dict) -> dict:
    """Whether the order is inside the return window. Unchanged from the raw build."""
    if order.get("status") not in ["delivered", "delivered_unconfirmed"]:
        return {
            "eligible": False,
            "days_remaining": 0,
            "window_closes": None,
            "reason": f"Order status is '{order['status']}'. "
                      f"Only delivered orders are eligible for return.",
        }

    if not order.get("actual_delivery"):
        return {
            "eligible": False,
            "days_remaining": 0,
            "window_closes": None,
            "reason": "No confirmed delivery date on record.",
        }

    delivered = datetime.strptime(order["actual_delivery"], "%Y-%m-%d")
    window_closes = delivered + timedelta(days=get_policy("return_window_days"))
    days_remaining = (window_closes - REFERENCE_DATE).days

    if days_remaining > 0:
        return {
            "eligible": True,
            "days_remaining": days_remaining,
            "window_closes": window_closes.strftime("%Y-%m-%d"),
            "reason": f"Within return window. {days_remaining} day(s) remaining.",
        }
    return {
        "eligible": False,
        "days_remaining": 0,
        "window_closes": window_closes.strftime("%Y-%m-%d"),
        "reason": f"Return window closed on {window_closes.strftime('%Y-%m-%d')}.",
    }


def returns_refunds(state: ChatState) -> dict:
    order_id = state["resolution"]["order_id"]
    order = lookup_order_by_id(order_id)
    return_record = lookup_return_by_order(order_id)
    eligibility = check_return_eligibility(order)

    facts: dict = {
        "order": order,
        "return_eligibility": eligibility,
        "return_record": return_record,
        "transactions": find_transactions_by_order(order_id),
    }
    triggers: list[EscalationTrigger] = []

    if eligibility["eligible"] and (
        eligibility["days_remaining"] < get_policy("return_window_closing_days")
    ):
        triggers.append("return_window_closing")

    if return_record and return_record.get("status") == "pending_review":
        days_pending = _days_since(return_record["return_requested_date"])
        facts["days_pending_review"] = days_pending
        if days_pending > get_policy("return_pending_escalation_days"):
            triggers.append("return_pending_10_days")

    return _finish(state, "returns_refunds", order_id, facts, triggers, RETURNS_SYSTEM)


# ---------------------------------------------------------------------------
# 6. Billing and Payments
# Analysis helpers unchanged from the raw build, except thresholds now come
# from STORE_POLICIES.
# ---------------------------------------------------------------------------

def detect_duplicate_charges(transactions: list) -> list[dict]:
    """Same amount on the same date more than once; all but the first are duplicates."""
    groups = defaultdict(list)
    for txn in transactions:
        groups[(txn["amount"], txn["transaction_date"])].append(txn)
    return [txn for group in groups.values() if len(group) > 1 for txn in group[1:]]


def check_refund_delays(transactions: list) -> list[dict]:
    """Refunds processing longer than the standard window."""
    standard_days = get_policy("refund_standard_days")
    delayed = []
    for txn in transactions:
        if txn.get("refund_status") == "processing" and txn.get("refund_initiated"):
            days = _days_since(txn["refund_initiated"])
            if days > standard_days:
                delayed.append({
                    **txn,
                    "days_processing": days,
                    "days_overdue": days - standard_days,
                })
    return delayed


def check_wrong_amount(order: dict, transactions: list) -> dict | None:
    """A completed charge that differs from the order total."""
    for txn in transactions:
        if txn.get("status") == "completed":
            discrepancy = round(abs(txn["amount"] - order["total"]), 2)
            if discrepancy > 0.01:
                return {
                    "transaction_id": txn["transaction_id"],
                    "order_total": order["total"],
                    "amount_charged": txn["amount"],
                    "discrepancy": discrepancy,
                    "overcharged": txn["amount"] > order["total"],
                }
    return None


def check_partial_refund(transactions: list) -> dict | None:
    """A partial refund that may be in dispute."""
    for txn in transactions:
        if txn.get("refund_status") == "partial":
            refunded = txn.get("refund_amount", 0)
            return {
                "transaction_id": txn["transaction_id"],
                "original_charge": txn["amount"],
                "refund_issued": refunded,
                "balance": round(txn["amount"] - refunded, 2),
            }
    return None


def billing_payments(state: ChatState) -> dict:
    order_id = state["resolution"]["order_id"]
    order = lookup_order_by_id(order_id)
    transactions = find_transactions_by_order(order_id)

    duplicates = detect_duplicate_charges(transactions)
    delayed = check_refund_delays(transactions)
    wrong_amount = check_wrong_amount(order, transactions)

    facts: dict = {
        "order": order,
        "transactions": transactions,
        "analysis": {
            "duplicate_charges": duplicates,
            "delayed_refunds": delayed,
            "wrong_amount": wrong_amount,
            "partial_refund": check_partial_refund(transactions),
        },
    }
    triggers: list[EscalationTrigger] = []

    if len(duplicates) > get_policy("duplicate_escalation_count"):
        triggers.append("multiple_duplicates")
    if any(d["days_processing"] > get_policy("refund_delay_escalation_days") for d in delayed):
        triggers.append("refund_delay_10_days")
    if wrong_amount and wrong_amount["discrepancy"] > get_policy("wrong_amount_escalation"):
        triggers.append("wrong_amount_over_50")

    return _finish(state, "billing_payments", order_id, facts, triggers, BILLING_SYSTEM)