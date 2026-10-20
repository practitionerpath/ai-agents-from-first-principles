import json
from collections import defaultdict
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from tools import (
    extract_order_id,
    extract_email,
    extract_phone,
    lookup_order_by_id,
    find_orders_by_email,
    find_orders_by_phone,
    lookup_customer_by_email,
    find_transactions_by_order,
    lookup_return_by_order,
    get_policy
)

load_dotenv(find_dotenv())
client = OpenAI()

BASE_DATE = datetime(2024, 2, 1)


# ---------------------------------------------------------------------------
# BILLING ANALYSIS HELPERS
# Run in code before the LLM call. Results are passed as explicit context.
# ---------------------------------------------------------------------------

def detect_duplicate_charges(transactions: list) -> list[dict]:
    """
    Detect duplicate charges: same amount charged more than once on the same date.
    Returns suspected duplicates excluding the first (legitimate) transaction per group.
    """
    groups = defaultdict(list)
    for txn in transactions:
        key = (txn["amount"], txn["transaction_date"])
        groups[key].append(txn)

    duplicates = []
    for group in groups.values():
        if len(group) > 1:
            duplicates.extend(group[1:])
    return duplicates


def check_refund_delays(transactions: list) -> list[dict]:
    """
    Identify refunds that have been processing beyond the standard 5-day window.
    Returns delayed transactions with days_processing and days_overdue fields added.
    """
    standard_days = 5
    delayed       = []
    for txn in transactions:
        if txn.get("refund_status") == "processing" and txn.get("refund_initiated"):
            initiated      = datetime.strptime(txn["refund_initiated"], "%Y-%m-%d")
            days_processing = (BASE_DATE - initiated).days
            if days_processing > standard_days:
                delayed.append({
                    **txn,
                    "days_processing": days_processing,
                    "days_overdue":    days_processing - standard_days
                })
    return delayed


def check_wrong_amount(order: dict, transactions: list) -> dict | None:
    """
    Check whether any transaction amount differs from the order total.
    Returns discrepancy details or None if all amounts match.
    Note: does not flag duplicate charges — that is handled separately.
    """
    for txn in transactions:
        if txn.get("status") == "completed":
            discrepancy = round(abs(txn["amount"] - order["total"]), 2)
            if discrepancy > 0.01:
                return {
                    "transaction_id":  txn["transaction_id"],
                    "order_total":     order["total"],
                    "amount_charged":  txn["amount"],
                    "discrepancy":     discrepancy,
                    "overcharged":     txn["amount"] > order["total"],
                    "notes":           txn.get("notes", "")
                }
    return None


def check_partial_refund_dispute(transactions: list, return_record: dict | None) -> dict | None:
    """
    Detect whether a partial refund was issued and may be in dispute.
    Returns dispute details or None if no partial refund exists.
    """
    for txn in transactions:
        if txn.get("refund_status") == "partial":
            original_amount = txn["amount"]
            refund_amount   = txn.get("refund_amount", 0)
            return {
                "transaction_id":  txn["transaction_id"],
                "original_charge": original_amount,
                "refund_issued":   refund_amount,
                "balance":         round(original_amount - refund_amount, 2),
                "notes":           txn.get("notes", "")
            }
    return None


def format_billing_context(order: dict, customer: dict, transactions: list,
                            return_record: dict | None,
                            duplicates: list, delayed_refunds: list,
                            wrong_amount: dict | None,
                            partial_dispute: dict | None) -> str:
    """
    Format all billing-relevant data into a readable block for the LLM prompt.
    """
    items_str = ", ".join([
        item.get("name", "Unknown") for item in order.get("items", [])
    ])

    txn_lines = []
    for txn in transactions:
        refund_info = ""
        if txn.get("refund_status"):
            refund_info = f" | refund: {txn['refund_status']}"
            if txn.get("refund_amount"):
                refund_info += f" ${txn['refund_amount']:.2f}"
            if txn.get("refund_initiated"):
                refund_info += f" (initiated {txn['refund_initiated']})"
        dup_flag = " *** SUSPECTED DUPLICATE ***" if txn in duplicates else ""
        txn_lines.append(
            f"  {txn['transaction_id']}: ${txn['amount']:.2f} "
            f"on {txn['transaction_date']}{refund_info}{dup_flag}"
        )

    duplicate_str = (
        f"DETECTED — {len(duplicates)} duplicate charge(s) found"
        if duplicates else "None detected"
    )

    delayed_str = "None"
    if delayed_refunds:
        delayed_str = "\n".join([
            f"  {d['transaction_id']}: ${d['refund_amount']:.2f} refund — "
            f"{d['days_processing']} days processing ({d['days_overdue']} days overdue)"
            for d in delayed_refunds
        ])

    wrong_str = "None detected"
    if wrong_amount:
        direction = "overcharged" if wrong_amount["overcharged"] else "undercharged"
        wrong_str = (
            f"Order total: ${wrong_amount['order_total']:.2f} | "
            f"Charged: ${wrong_amount['amount_charged']:.2f} | "
            f"Discrepancy: ${wrong_amount['discrepancy']:.2f} ({direction})"
        )
        if wrong_amount.get("notes"):
            wrong_str += f"\n  Notes: {wrong_amount['notes']}"

    partial_str = "None"
    if partial_dispute:
        partial_str = (
            f"Original charge: ${partial_dispute['original_charge']:.2f} | "
            f"Refund issued: ${partial_dispute['refund_issued']:.2f} | "
            f"Balance remaining: ${partial_dispute['balance']:.2f}"
        )
        if partial_dispute.get("notes"):
            partial_str += f"\n  Notes: {partial_dispute['notes']}"

    return_str = "No return on file"
    if return_record:
        return_str = (
            f"{return_record['return_id']} | "
            f"Status: {return_record['status']} | "
            f"Refund: ${return_record.get('refund_amount', 0):.2f} "
            f"({return_record.get('refund_status', 'N/A')})"
        )

    return f"""Customer: {customer.get('name', 'Unknown')} \
({customer.get('account_tier', 'standard').capitalize()} \
member since {customer.get('member_since', 'N/A')})

Order ID     : {order['order_id']}
Items        : {items_str}
Order total  : ${order['total']:.2f}
Order status : {order['status']}
Order date   : {order['order_date']}
Notes        : {order.get('notes', 'None')}

Transactions:
{chr(10).join(txn_lines) if txn_lines else '  None found'}

Billing analysis:
  Duplicate charges  : {duplicate_str}
  Delayed refunds    : {delayed_str}
  Wrong amount       : {wrong_str}
  Partial refund     : {partial_str}

Return record        : {return_str}

Store policies:
  Refund timeline          : {get_policy('refund_timeline_business_days')} business days
  Duplicate charge resolution: {get_policy('duplicate_charge_resolution')}
  High value threshold     : ${get_policy('high_value_threshold'):.2f}"""


def run_billing_payments(payload: dict) -> dict:
    """
    Billing and Payments specialist.
    Receives: {customer_message, prior_context}
    Returns:  {assessment, recommended_action,
               needs_human, needs_clarification, clarification_question}
    """
    customer_message = payload["customer_message"]
    prior_context    = payload.get("prior_context", "")

    # --- Step 1: Extract identifying information ---
    order_id = extract_order_id(customer_message)
    email    = extract_email(customer_message)
    phone    = extract_phone(customer_message)

    print(f"[Billing & Payments] Extracted — order_id: {order_id}, "
          f"email: {email}, phone: {phone}")

    # --- Step 2: Attempt order lookup ---
    order    = None
    customer = None

    if order_id:
        order = lookup_order_by_id(order_id)
        if order:
            customer = lookup_customer_by_email(order["customer_email"])

    if not order and email:
        orders = find_orders_by_email(email)
        if len(orders) == 1:
            order    = orders[0]
            customer = lookup_customer_by_email(email)
        elif len(orders) > 1:
            return {
                "assessment":             "Multiple orders found on this account.",
                "recommended_action":     "Ask customer to specify order number.",
                "needs_human":            False,
                "needs_clarification":    True,
                "clarification_question": (
                    "We found multiple orders on your account. "
                    "Could you provide your order number so we can "
                    "review the correct billing details?"
                )
            }

    if not order and phone:
        orders = find_orders_by_phone(phone)
        if len(orders) == 1:
            order    = orders[0]
            customer = lookup_customer_by_email(order["customer_email"])
        elif len(orders) > 1:
            return {
                "assessment":             "Multiple orders found on this account.",
                "recommended_action":     "Ask customer to specify order number.",
                "needs_human":            False,
                "needs_clarification":    True,
                "clarification_question": (
                    "We found multiple orders linked to your phone number. "
                    "Could you share your order number so we can check "
                    "the billing details for the right one?"
                )
            }

    # --- Step 3: No identifying information found ---
    if not order:
        return {
            "assessment":             "Unable to locate an order with the information provided.",
            "recommended_action":     "Request identifying information from customer.",
            "needs_human":            False,
            "needs_clarification":    True,
            "clarification_question": (
                "I'd be happy to help with your billing concern. "
                "Could you please provide your order number or the "
                "email address on your account?"
            )
        }

    # --- Step 4: Run billing analysis in code ---
    transactions    = find_transactions_by_order(order["order_id"])
    return_record   = lookup_return_by_order(order["order_id"])
    duplicates      = detect_duplicate_charges(transactions)
    delayed_refunds = check_refund_delays(transactions)
    wrong_amount    = check_wrong_amount(order, transactions)
    partial_dispute = check_partial_refund_dispute(transactions, return_record)
    high_value      = order["total"] >= get_policy("high_value_threshold")

    # Refund delayed more than 10 days warrants human review
    severely_delayed = any(d["days_processing"] > 10 for d in delayed_refunds)

    billing_context = format_billing_context(
        order, customer or {}, transactions, return_record,
        duplicates, delayed_refunds, wrong_amount, partial_dispute
    )

    # --- Step 5: LLM assessment ---
    prior_section = (
        f"\nContext from other specialists already run:\n{prior_context}"
        if prior_context else ""
    )

    system_prompt = """You are a billing and payments specialist for an e-commerce company.

You handle: incorrect charges, duplicate billing, payment failures,
refund status inquiries, and charge disputes.

Use the billing analysis provided. Do not re-derive what is already computed.
Apply policies precisely.

Resolution guidelines:
- Duplicate charge: confirm the duplicate, initiate refund within 24 hours
- Delayed refund: acknowledge delay, confirm it is processing, escalate
  internally if beyond 10 days
- Wrong amount charged: confirm the discrepancy, initiate refund for the difference
- Partial refund dispute: assess whether the partial amount was correctly
  applied per policy (wrong item or defective = full refund, change of mind = 10% fee)

Return ONLY a JSON object in this exact format:
{
    "assessment": "specific description of the billing issue based on the data",
    "recommended_action": "specific resolution steps for this exact situation",
    "needs_human": true | false,
    "needs_clarification": false,
    "clarification_question": ""
}

Set needs_human to true if:
- Order total is $500 or above
- More than one duplicate charge found
- Refund has been processing for more than 10 days
- Wrong amount discrepancy exceeds $50"""

    user_prompt = f"""Customer message:
{customer_message}

Billing data:
{billing_context}

High value order (>= $500) : {high_value}
Severely delayed refund (> 10 days): {severely_delayed}
Duplicate charges detected : {len(duplicates) > 0}
Wrong amount detected      : {wrong_amount is not None}{prior_section}

Assess the billing issue and recommend a resolution."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )

    try:
        result = json.loads(response.choices[0].message.content)
        if "assessment" not in result or "recommended_action" not in result:
            raise ValueError("Missing required fields")

        result.setdefault("needs_human",           high_value or severely_delayed)
        result.setdefault("needs_clarification",   False)
        result.setdefault("clarification_question", "")

        # Code-level overrides
        if high_value:
            result["needs_human"] = True
        if severely_delayed:
            result["needs_human"] = True

        print(f"[Billing & Payments] Assessment complete. "
              f"duplicates: {len(duplicates)}, "
              f"delayed: {len(delayed_refunds)}, "
              f"wrong_amount: {wrong_amount is not None}, "
              f"needs_human: {result['needs_human']}")
        return result

    except (json.JSONDecodeError, ValueError) as e:
        print(f"[Billing & Payments] Failed to parse LLM response: {e}")
        return {
            "assessment":             "Unable to complete billing assessment due to a processing error.",
            "recommended_action":     "Route to human agent for manual review.",
            "needs_human":            True,
            "needs_clarification":    False,
            "clarification_question": ""
        }