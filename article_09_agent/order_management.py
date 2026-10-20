import json
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
    get_contact_history,
    get_policy
)

load_dotenv(find_dotenv())
client = OpenAI()


def format_order_context(order: dict, customer: dict,
                         transactions: list, return_record: dict | None) -> str:
    """
    Format order data into a readable block for the LLM prompt.
    """
    items_str = ", ".join([
        f"{item.get('name', 'Unknown')} "
        f"{'x' + str(item['qty']) if item.get('qty', 1) > 1 else ''}"
        for item in order.get("items", [])
    ])

    txn_lines = []
    for txn in transactions:
        refund_info = ""
        if txn.get("refund_status"):
            refund_info = (
                f" | Refund: {txn['refund_status']} "
                f"(${txn.get('refund_amount', 0):.2f})"
            )
        txn_lines.append(
            f"  {txn['transaction_id']}: ${txn['amount']:.2f} "
            f"on {txn['transaction_date']}{refund_info}"
        )

    return_str = "None on file"
    if return_record:
        return_str = (
            f"{return_record['return_id']} | "
            f"Status: {return_record['status']} | "
            f"Reason: {return_record['reason']}"
        )

    return f"""Customer: {customer.get('name', 'Unknown')} \
({customer.get('account_tier', 'standard').capitalize()} member since \
{customer.get('member_since', 'N/A')})

Order ID     : {order['order_id']}
Status       : {order['status']}
Items        : {items_str}
Total        : ${order['total']:.2f}
Order date   : {order['order_date']}
Expected     : {order.get('expected_delivery', 'N/A')}
Delivered    : {order.get('actual_delivery', 'Not yet delivered')}
Carrier      : {order.get('carrier', 'N/A')}
Tracking     : {order.get('tracking_number', 'N/A')}
Notes        : {order.get('notes', 'None')}

Transactions:
{chr(10).join(txn_lines) if txn_lines else '  None found'}

Return record: {return_str}"""


def run_order_management(payload: dict) -> dict:
    """
    Order Management specialist.
    Receives: {customer_message, prior_context}
    Returns:  {assessment, recommended_action,
               needs_human, needs_clarification, clarification_question}
    """
    customer_message = payload["customer_message"]
    prior_context    = payload.get("prior_context", "")

    # --- Step 1: Extract identifying information from the message ---
    order_id = extract_order_id(customer_message)
    email    = extract_email(customer_message)
    phone    = extract_phone(customer_message)

    print(f"[Order Management] Extracted — order_id: {order_id}, "
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
            # Multiple orders found: ask customer to specify
            return {
                "assessment":            "Multiple orders found on this account.",
                "recommended_action":    "Ask customer to provide their order number.",
                "needs_human":           False,
                "needs_clarification":   True,
                "clarification_question": (
                    "We found multiple orders on your account. "
                    "Could you please provide your order number "
                    "so we can look into the right one for you?"
                )
            }

    if not order and phone:
        orders = find_orders_by_phone(phone)
        if len(orders) == 1:
            order    = orders[0]
            customer = lookup_customer_by_email(order["customer_email"])
        elif len(orders) > 1:
            return {
                "assessment":            "Multiple orders found on this account.",
                "recommended_action":    "Ask customer to provide their order number.",
                "needs_human":           False,
                "needs_clarification":   True,
                "clarification_question": (
                    "We found multiple orders linked to your phone number. "
                    "Could you please share your order number "
                    "so we can pull up the right one?"
                )
            }

    # --- Step 3: No identifying information found ---
    if not order:
        return {
            "assessment":            "Unable to locate an order with the information provided.",
            "recommended_action":    "Request identifying information from customer.",
            "needs_human":           False,
            "needs_clarification":   True,
            "clarification_question": (
                "I'd be happy to help with your order. "
                "Could you please provide your order number, "
                "the email address on your account, or your phone number?"
            )
        }

    # --- Step 4: Gather supporting data ---
    transactions  = find_transactions_by_order(order["order_id"])
    return_record = lookup_return_by_order(order["order_id"])
    order_context = format_order_context(order, customer or {}, transactions, return_record)
    high_value    = order["total"] >= get_policy("high_value_threshold")

    # --- Step 5: LLM assessment ---
    prior_section = (
        f"\nContext from other specialists already run:\n{prior_context}"
        if prior_context else ""
    )

    system_prompt = """You are an order management specialist for an e-commerce company.

You handle: delivery issues, wrong or missing items, damaged goods,
tracking problems, and order cancellations.

Resolution policies you apply:
- Orders not arrived within 7 days of expected delivery: open carrier investigation
- Wrong item received: customer keeps wrong item, correct item reshipped at no cost
- Damaged on arrival: photo evidence requested, replacement or refund within 2 business days
- Delivered per carrier but customer says not received: open carrier investigation,
  resolution within 5 business days. If warehouse also shows unconfirmed, flag for human review.
- Cancelled orders: refund processed within 3-5 business days

Assess the order issue specifically and precisely.
Do not give generic advice. Use the order data provided.

Return ONLY a JSON object in this exact format:
{
    "assessment": "specific description of the order issue based on the data",
    "recommended_action": "specific resolution steps for this exact situation",
    "needs_human": true | false,
    "needs_clarification": false,
    "clarification_question": ""
}

Set needs_human to true if:
- Order total is $500 or above
- Order status is delivered_unconfirmed (carrier and warehouse conflict)
- Order is more than 14 days overdue with no carrier update
- Situation is too ambiguous for policy to apply cleanly"""

    user_prompt = f"""Customer message:
{customer_message}

Order data:
{order_context}

High value order (>= $500): {high_value}{prior_section}

Assess the order issue and recommend a resolution."""

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

        # Ensure all expected fields are present
        result.setdefault("needs_human",           high_value)
        result.setdefault("needs_clarification",   False)
        result.setdefault("clarification_question", "")

        # Code-level override: always flag high-value orders for human review
        # regardless of what the LLM decides.
        if high_value:
            result["needs_human"] = True

        print(f"[Order Management] Assessment complete. "
              f"needs_human: {result['needs_human']}")
        return result

    except (json.JSONDecodeError, ValueError) as e:
        print(f"[Order Management] Failed to parse LLM response: {e}")
        return {
            "assessment":            "Unable to complete order assessment due to a processing error.",
            "recommended_action":    "Route to human agent for manual review.",
            "needs_human":           True,
            "needs_clarification":   False,
            "clarification_question": ""
        }