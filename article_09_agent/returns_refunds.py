import json
from datetime import datetime, timedelta
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

# Base date for all return window calculations.
# Matches the reference date used across all mock data records.
# In production this would be replaced by datetime.now().
BASE_DATE = datetime(2024, 2, 1)


def check_return_eligibility(order: dict) -> dict:
    """
    Determine whether an order is within the return window.
    Returns a dict with eligibility, days remaining, and window close date.
    """
    if order.get("status") not in ["delivered", "delivered_unconfirmed"]:
        return {
            "eligible": False,
            "days_remaining": 0,
            "window_closes": None,
            "reason": f"Order status is '{order['status']}'. "
                      f"Only delivered orders are eligible for return."
        }

    delivery_date_str = order.get("actual_delivery")
    if not delivery_date_str:
        return {
            "eligible": False,
            "days_remaining": 0,
            "window_closes": None,
            "reason": "No confirmed delivery date on record."
        }

    delivery_date  = datetime.strptime(delivery_date_str, "%Y-%m-%d")
    return_window  = get_policy("return_window_days")
    window_closes  = delivery_date + timedelta(days=return_window)
    days_remaining = (window_closes - BASE_DATE).days

    if days_remaining > 0:
        return {
            "eligible":       True,
            "days_remaining": days_remaining,
            "window_closes":  window_closes.strftime("%Y-%m-%d"),
            "reason":         f"Within return window. {days_remaining} day(s) remaining."
        }
    else:
        return {
            "eligible":       False,
            "days_remaining": 0,
            "window_closes":  window_closes.strftime("%Y-%m-%d"),
            "reason":         f"Return window closed on {window_closes.strftime('%Y-%m-%d')}."
        }


def format_returns_context(order: dict, customer: dict, transactions: list,
                           return_record: dict | None,
                           eligibility: dict) -> str:
    """
    Format all return-relevant data into a readable block for the LLM prompt.
    """
    items_str = ", ".join([
        f"{item.get('name', 'Unknown')}"
        for item in order.get("items", [])
    ])

    txn_lines = []
    for txn in transactions:
        refund_info = ""
        if txn.get("refund_status"):
            refund_info = (
                f" | Refund: {txn['refund_status']} "
                f"(${txn.get('refund_amount', 0):.2f})"
                + (f" on {txn['refund_date']}" if txn.get("refund_date") else "")
            )
        txn_lines.append(
            f"  {txn['transaction_id']}: ${txn['amount']:.2f} "
            f"charged on {txn['transaction_date']}{refund_info}"
        )

    return_str = "No return on file"
    if return_record:
        return_str = (
            f"Return ID     : {return_record['return_id']}\n"
            f"  Status        : {return_record['status']}\n"
            f"  Reason        : {return_record['reason']}\n"
            f"  Requested     : {return_record.get('return_requested_date', 'N/A')}\n"
            f"  Refund amount : ${return_record.get('refund_amount', 0):.2f}\n"
            f"  Refund status : {return_record.get('refund_status', 'N/A')}\n"
            f"  Notes         : {return_record.get('notes', 'None')}"
        )

    return f"""Customer: {customer.get('name', 'Unknown')} \
({customer.get('account_tier', 'standard').capitalize()} \
member since {customer.get('member_since', 'N/A')})

Order ID     : {order['order_id']}
Status       : {order['status']}
Items        : {items_str}
Total        : ${order['total']:.2f}
Delivered    : {order.get('actual_delivery', 'Not yet delivered')}
Notes        : {order.get('notes', 'None')}

Return eligibility:
  Eligible      : {eligibility['eligible']}
  Days remaining: {eligibility['days_remaining']}
  Window closes : {eligibility.get('window_closes', 'N/A')}
  Reason        : {eligibility['reason']}

Transactions:
{chr(10).join(txn_lines) if txn_lines else '  None found'}

Existing return record:
  {return_str}

Store policies:
  Return window      : {get_policy('return_window_days')} days from delivery
  Restocking fee     : {int(get_policy('restocking_fee_standard') * 100)}% \
(change of mind) | 0% (wrong item or defective)
  Refund timeline    : {get_policy('refund_timeline_business_days')} business days
  High value threshold: ${get_policy('high_value_threshold'):.2f} \
(requires human review)"""


def run_returns_refunds(payload: dict) -> dict:
    """
    Returns and Refunds specialist.
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

    print(f"[Returns & Refunds] Extracted — order_id: {order_id}, "
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
                    "Could you please provide your order number "
                    "so we can check the return details for the right one?"
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
                    "the return details for the correct order?"
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
                "I'd be happy to help with your return or refund. "
                "Could you please provide your order number or the "
                "email address associated with your account?"
            )
        }

    # --- Step 4: Gather supporting data ---
    transactions   = find_transactions_by_order(order["order_id"])
    return_record  = lookup_return_by_order(order["order_id"])
    eligibility    = check_return_eligibility(order)
    high_value     = order["total"] >= get_policy("high_value_threshold")

    # Pending return older than 10 days warrants human review
    pending_too_long = False
    if return_record and return_record.get("status") == "pending_review":
        requested_str = return_record.get("return_requested_date")
        if requested_str:
            requested_date   = datetime.strptime(requested_str, "%Y-%m-%d")
            days_pending     = (BASE_DATE - requested_date).days
            pending_too_long = days_pending > 10

    returns_context = format_returns_context(
        order, customer or {}, transactions, return_record, eligibility
    )

    # --- Step 5: LLM assessment ---
    prior_section = (
        f"\nContext from other specialists already run:\n{prior_context}"
        if prior_context else ""
    )

    system_prompt = """You are a returns and refunds specialist for an e-commerce company.

You handle: return requests, exchange requests, refund eligibility checks,
return status inquiries, and store policy questions.

Use the order data, eligibility check, and existing return record provided.
Apply store policies precisely. Do not guess or generalise.

Key policy rules:
- Returns are only accepted within the return window from delivery date
- Wrong item or defective item: no restocking fee, full refund or replacement
- Change of mind: 10% restocking fee applies
- If a return is already approved and refund is delayed beyond 5 business days,
  acknowledge the delay and confirm the refund is being processed
- If a partial refund was issued and customer disputes it, assess whether the
  partial amount was correctly applied per policy

Return ONLY a JSON object in this exact format:
{
    "assessment": "specific description of the return or refund situation",
    "recommended_action": "specific next steps based on policy and order data",
    "needs_human": true | false,
    "needs_clarification": false,
    "clarification_question": ""
}

Set needs_human to true if:
- Order total is $500 or above
- Return window has fewer than 3 days remaining (borderline case)
- A return request has been pending review for more than 10 days
- Customer disputes a partial refund and policy is ambiguous about the amount"""

    user_prompt = f"""Customer message:
{customer_message}

Returns and refund data:
{returns_context}

High value order (>= $500): {high_value}
Return pending more than 10 days: {pending_too_long}{prior_section}

Assess the return or refund situation and recommend a resolution."""

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

        result.setdefault("needs_human",           high_value or pending_too_long)
        result.setdefault("needs_clarification",   False)
        result.setdefault("clarification_question", "")

        # Code-level overrides
        if high_value:
            result["needs_human"] = True
        if pending_too_long:
            result["needs_human"] = True

        print(f"[Returns & Refunds] Assessment complete. "
              f"needs_human: {result['needs_human']}, "
              f"eligible: {eligibility['eligible']}")
        return result

    except (json.JSONDecodeError, ValueError) as e:
        print(f"[Returns & Refunds] Failed to parse LLM response: {e}")
        return {
            "assessment":             "Unable to complete returns assessment due to a processing error.",
            "recommended_action":     "Route to human agent for manual review.",
            "needs_human":            True,
            "needs_clarification":    False,
            "clarification_question": ""
        }