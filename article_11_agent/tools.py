import re
from mock_data import (
    CUSTOMERS,
    CUSTOMERS_BY_PHONE,
    ORDERS,
    TRANSACTIONS,
    RETURN_RECORDS,
    CONTACT_HISTORY,
    STORE_POLICIES
)


# ---------------------------------------------------------------------------
# MESSAGE PARSING
# Extract identifying information from a customer message before lookup.
# ---------------------------------------------------------------------------

def extract_order_id(message: str) -> str | None:
    """
    Extract an order ID from a customer message.
    Matches: ORD-1001, order #1001, order 1001, #1001
    Returns the normalised ORD-XXXX format, or None if not found.
    """
    pattern = (
        r'\b(ORD-\d{3,6})\b'           # explicit ORD-1001 format
        r'|order\s*#?\s*(\d{3,6})\b'   # order #1001 or order 1001
        r'|#(\d{3,6})\b'               # standalone #1001
    )
    match = re.search(pattern, message, re.IGNORECASE)
    if not match:
        return None
    matched = next(g for g in match.groups() if g)
    return matched if matched.upper().startswith("ORD-") else f"ORD-{matched}"


def extract_email(message: str) -> str | None:
    """
    Extract an email address from a customer message.
    Returns the first match found, or None.
    """
    pattern = r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
    match = re.search(pattern, message)
    return match.group(0).lower() if match else None


def extract_phone(message: str) -> str | None:
    """
    Extract a phone number from a customer message.
    Matches formats: 617-555-0101, (617) 555-0101, 6175550101
    Returns the first match found, or None.
    """
    pattern = r'\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}'
    match = re.search(pattern, message)
    if not match:
        return None
    # Normalise to XXX-XXX-XXXX format
    digits = re.sub(r'\D', '', match.group(0))
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}" if len(digits) == 10 else None


# ---------------------------------------------------------------------------
# CUSTOMER LOOKUP
# ---------------------------------------------------------------------------

def lookup_customer_by_email(email: str) -> dict | None:
    """
    Look up a customer by email address.
    Returns the customer record or None if not found.
    """
    result = CUSTOMERS.get(email.lower().strip())
    if not result:
        print(f"[tools] No customer found for email: {email}")
    return result


def lookup_customer_by_phone(phone: str) -> dict | None:
    """
    Look up a customer by phone number.
    Returns the customer record or None if not found.
    """
    result = CUSTOMERS_BY_PHONE.get(phone.strip())
    if not result:
        print(f"[tools] No customer found for phone: {phone}")
    return result


# ---------------------------------------------------------------------------
# ORDER LOOKUP
# ---------------------------------------------------------------------------

def lookup_order_by_id(order_id: str) -> dict | None:
    """
    Look up an order by order ID.
    Returns the order record or None if not found.
    """
    result = ORDERS.get(order_id.upper().strip())
    if not result:
        print(f"[tools] No order found for ID: {order_id}")
    return result


def find_orders_by_email(email: str) -> list[dict]:
    """
    Find all orders associated with a customer email address.
    Returns a list of matching order records, empty list if none found.
    """
    results = [
        order for order in ORDERS.values()
        if order["customer_email"].lower() == email.lower().strip()
    ]
    if not results:
        print(f"[tools] No orders found for email: {email}")
    return results


def find_orders_by_phone(phone: str) -> list[dict]:
    """
    Find all orders associated with a customer phone number.
    Looks up the customer by phone first, then finds their orders by email.
    Returns a list of matching order records, empty list if none found.
    """
    customer = lookup_customer_by_phone(phone)
    if not customer:
        return []
    return find_orders_by_email(customer["email"])


# ---------------------------------------------------------------------------
# TRANSACTION LOOKUP
# ---------------------------------------------------------------------------

def find_transactions_by_order(order_id: str) -> list[dict]:
    """
    Find all transactions associated with an order ID.
    Returns a list of matching transaction records, empty list if none found.
    """
    results = [
        txn for txn in TRANSACTIONS.values()
        if txn["order_id"].upper() == order_id.upper().strip()
    ]
    if not results:
        print(f"[tools] No transactions found for order: {order_id}")
    return results


def find_transactions_by_email(email: str) -> list[dict]:
    """
    Find all transactions associated with a customer email address.
    Returns a list of matching transaction records, empty list if none found.
    """
    results = [
        txn for txn in TRANSACTIONS.values()
        if txn["customer_email"].lower() == email.lower().strip()
    ]
    if not results:
        print(f"[tools] No transactions found for email: {email}")
    return results


# ---------------------------------------------------------------------------
# RETURN RECORD LOOKUP
# ---------------------------------------------------------------------------

def lookup_return_by_order(order_id: str) -> dict | None:
    """
    Find the return record associated with an order ID.
    Returns the return record or None if no return has been filed.
    """
    for record in RETURN_RECORDS.values():
        if record["order_id"].upper() == order_id.upper().strip():
            return record
    print(f"[tools] No return record found for order: {order_id}")
    return None


# ---------------------------------------------------------------------------
# CONTACT HISTORY LOOKUP
# ---------------------------------------------------------------------------

def get_contact_history(email: str) -> dict | None:
    """
    Retrieve the contact history for a customer by email address.
    Returns the contact history record or None if not found.
    """
    result = CONTACT_HISTORY.get(email.lower().strip())
    if not result:
        print(f"[tools] No contact history found for email: {email}")
    return result


# ---------------------------------------------------------------------------
# POLICY LOOKUP
# ---------------------------------------------------------------------------

def get_policy(key: str):
    """
    Retrieve a specific store policy value by key.
    Returns the value or None if the key does not exist.

    Available keys:
      return_window_days, return_window_basis,
      restocking_fee_standard, restocking_fee_defective,
      refund_timeline_business_days, duplicate_charge_resolution,
      high_value_threshold, escalation_contact_threshold,
      free_returns_threshold
    """
    result = STORE_POLICIES.get(key)
    if result is None:
        print(f"[tools] No policy found for key: {key}")
    return result