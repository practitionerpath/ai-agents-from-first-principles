# mock_data.py
# All records are mock data representing a small e-commerce store.
# In production each dict would be replaced by a database or API call.
# Base date for all relative references: 2024-02-01

# ---------------------------------------------------------------------------
# STORE POLICIES
# Referenced by returns_refunds.py and billing_payments.py specialists.
# ---------------------------------------------------------------------------

STORE_POLICIES = {
    "return_window_days":            30,
    "return_window_basis":           "delivery_date",
    "restocking_fee_standard":       0.10,   # 10% for change-of-mind returns
    "restocking_fee_defective":      0.00,   # no fee for wrong item or defective
    "refund_timeline_business_days": "3-5",
    "duplicate_charge_resolution":   "24 hours",
    "high_value_threshold":          500.00, # orders above this trigger needs_human
    "escalation_contact_threshold":  2,      # prior contacts >= this triggers escalation
    "free_returns_threshold":        500.00  # above this amount, human review required
}


# ---------------------------------------------------------------------------
# CUSTOMERS
# Six customers. Five base scenarios plus one for Article 11 failure modes.
# Lookup by email (primary) or phone (secondary).
# ---------------------------------------------------------------------------

CUSTOMERS = {
    "sarah.jones@email.com": {
        "customer_id":   "CUST-001",
        "name":          "Sarah Jones",
        "email":         "sarah.jones@email.com",
        "phone":         "617-555-0101",
        "account_tier":  "standard",
        "member_since":  "2022-03-15"
    },
    "mike.chen@email.com": {
        "customer_id":   "CUST-002",
        "name":          "Mike Chen",
        "email":         "mike.chen@email.com",
        "phone":         "617-555-0202",
        "account_tier":  "premium",
        "member_since":  "2021-06-01"
    },
    "priya.patel@email.com": {
        "customer_id":   "CUST-003",
        "name":          "Priya Patel",
        "email":         "priya.patel@email.com",
        "phone":         "617-555-0303",
        "account_tier":  "standard",
        "member_since":  "2023-01-10"
    },
    "james.wilson@email.com": {
        "customer_id":   "CUST-004",
        "name":          "James Wilson",
        "email":         "james.wilson@email.com",
        "phone":         "617-555-0404",
        "account_tier":  "premium",
        "member_since":  "2020-09-22"
    },
    "lisa.martin@email.com": {
        "customer_id":   "CUST-005",
        "name":          "Lisa Martin",
        "email":         "lisa.martin@email.com",
        "phone":         "617-555-0505",
        "account_tier":  "standard",
        "member_since":  "2023-07-18"
    },
    # CUST-006: added for Article 11 failure mode scenarios.
    # Kept separate from base customers to avoid contaminating clean scenarios.
    "david.kim@email.com": {
        "customer_id":   "CUST-006",
        "name":          "David Kim",
        "email":         "david.kim@email.com",
        "phone":         "617-555-0606",
        "account_tier":  "standard",
        "member_since":  "2024-01-05"
    }
}

# Phone number lookup index
CUSTOMERS_BY_PHONE = {
    c["phone"]: c for c in CUSTOMERS.values()
}


# ---------------------------------------------------------------------------
# ORDERS
# Ten orders covering all routing scenarios across Articles 8-12.
# ---------------------------------------------------------------------------

ORDERS = {

    # --- Simple and compound scenarios (Article 8) ---

    "ORD-1001": {
        # Scenario: wrong item received + pending return
        # Routing: order_management + returns_refunds
        "order_id":          "ORD-1001",
        "customer_email":    "sarah.jones@email.com",
        "status":            "delivered",
        "items":             [{"name": "Blue Denim Jacket", "size": "M", "qty": 1}],
        "total":             89.99,
        "order_date":        "2024-01-10",
        "expected_delivery": "2024-01-15",
        "actual_delivery":   "2024-01-15",
        "carrier":           "FedEx",
        "tracking_number":   "FX123456789",
        "notes":             "Customer reports receiving Size L instead of Size M"
    },

    "ORD-1002": {
        # Scenario: simple in-transit order inquiry, no issues
        # Routing: order_management only
        "order_id":          "ORD-1002",
        "customer_email":    "mike.chen@email.com",
        "status":            "in_transit",
        "items":             [
            {"name": "Wireless Headphones", "qty": 1},
            {"name": "Phone Case",          "qty": 1}
        ],
        "total":             149.99,
        "order_date":        "2024-01-25",
        "expected_delivery": "2024-02-03",
        "actual_delivery":   None,
        "carrier":           "UPS",
        "tracking_number":   "UP987654321",
        "notes":             None
    },

    "ORD-1003": {
        # Scenario: processing order with duplicate charge in transactions
        # Routing: billing_payments only
        "order_id":          "ORD-1003",
        "customer_email":    "priya.patel@email.com",
        "status":            "processing",
        "items":             [{"name": "Running Shoes", "size": "8", "qty": 1}],
        "total":             120.00,
        "order_date":        "2024-01-30",
        "expected_delivery": "2024-02-05",
        "actual_delivery":   None,
        "carrier":           None,
        "tracking_number":   None,
        "notes":             "Order not yet shipped. Two charges appear in transactions."
    },

    "ORD-1004": {
        # Scenario: defective item, return approved but refund stuck 8 days
        # Customer has 3 prior contacts. Strong escalation trigger.
        # Routing: returns_refunds + billing_payments + escalation
        "order_id":          "ORD-1004",
        "customer_email":    "james.wilson@email.com",
        "status":            "delivered",
        "items":             [{"name": "Coffee Maker", "model": "BrewMaster 3000", "qty": 1}],
        "total":             75.00,
        "order_date":        "2024-01-05",
        "expected_delivery": "2024-01-10",
        "actual_delivery":   "2024-01-10",
        "carrier":           "USPS",
        "tracking_number":   "US111222333",
        "notes":             "Item reported defective. Does not power on."
    },

    "ORD-1005": {
        # Scenario: cancelled order, refund not received after 11 days
        # Routing: billing_payments only
        "order_id":          "ORD-1005",
        "customer_email":    "lisa.martin@email.com",
        "status":            "cancelled",
        "items":             [
            {"name": "Yoga Mat",        "qty": 1},
            {"name": "Resistance Bands","qty": 1}
        ],
        "total":             55.00,
        "order_date":        "2024-01-20",
        "expected_delivery": None,
        "actual_delivery":   None,
        "carrier":           None,
        "tracking_number":   None,
        "notes":             "Cancelled by customer 2024-01-21. Refund not yet received."
    },

    "ORD-1006": {
        # Scenario: high-value order overdue, triggers needs_human
        # Routing: order_management only (needs_human: True due to $520 threshold)
        "order_id":          "ORD-1006",
        "customer_email":    "lisa.martin@email.com",
        "status":            "in_transit",
        "items":             [{"name": "Standing Desk", "qty": 1}],
        "total":             520.00,
        "order_date":        "2024-01-22",
        "expected_delivery": "2024-01-28",
        "actual_delivery":   None,
        "carrier":           "FedEx",
        "tracking_number":   "FX987654321",
        "notes":             "Expected 2024-01-28. No delivery update as of 2024-02-01 (4 days overdue)."
    },

    "ORD-1007": {
        # Scenario: partial refund dispute
        # $20 refund issued, customer expects full $45 refund
        # Routing: returns_refunds + billing_payments
        "order_id":          "ORD-1007",
        "customer_email":    "sarah.jones@email.com",
        "status":            "delivered",
        "items":             [{"name": "Laptop Stand", "qty": 1}],
        "total":             45.00,
        "order_date":        "2024-01-10",
        "expected_delivery": "2024-01-15",
        "actual_delivery":   "2024-01-15",
        "carrier":           "UPS",
        "tracking_number":   "UP111333555",
        "notes":             "Partial refund of $20 issued. Customer disputes, expects full refund."
    },

    "ORD-1008": {
        # Scenario: return window edge case
        # Delivered 2024-01-07, return window closes 2024-02-06 (5 days remaining)
        # Routing: returns_refunds only (tests policy boundary application)
        "order_id":          "ORD-1008",
        "customer_email":    "mike.chen@email.com",
        "status":            "delivered",
        "items":             [{"name": "Bluetooth Speaker", "qty": 1}],
        "total":             68.00,
        "order_date":        "2024-01-02",
        "expected_delivery": "2024-01-07",
        "actual_delivery":   "2024-01-07",
        "carrier":           "USPS",
        "tracking_number":   "US444555666",
        "notes":             "Delivered 2024-01-07. Return window closes 2024-02-06. 5 days remaining."
    },

    # --- Article 11 failure mode scenarios ---

    "ORD-1009": {
        # Scenario: ambiguous delivery status
        # Carrier marks delivered, warehouse shows pending confirmation
        # Customer says item not received
        # Failure mode: order_management produces wrong assessment based on carrier data alone
        "order_id":          "ORD-1009",
        "customer_email":    "david.kim@email.com",
        "status":            "delivered_unconfirmed",
        "items":             [{"name": "Kitchen Blender", "qty": 1}],
        "total":             95.00,
        "order_date":        "2024-01-25",
        "expected_delivery": "2024-01-30",
        "actual_delivery":   "2024-01-30",  # carrier says delivered
        "carrier":           "FedEx",
        "tracking_number":   "FX000111222",
        "notes":             "Carrier confirms delivery 2024-01-30. "
                             "Warehouse status: pending confirmation. "
                             "Customer reports item not received."
    },

    "ORD-1010": {
        # Scenario: wrong charge amount (subtle discrepancy)
        # Charged $89.99, sale price at order time was $79.99
        # Failure mode: billing specialist may miss the $10 discrepancy if not checking
        # order date vs. transaction amount carefully
        "order_id":          "ORD-1010",
        "customer_email":    "david.kim@email.com",
        "status":            "delivered",
        "items":             [{"name": "Smartwatch", "qty": 1}],
        "total":             79.99,          # correct price at time of order
        "order_date":        "2024-01-28",
        "expected_delivery": "2024-01-31",
        "actual_delivery":   "2024-01-31",
        "carrier":           "UPS",
        "tracking_number":   "UP222333444",
        "notes":             "Item was on sale at $79.99 when ordered. "
                             "Customer was charged $89.99 (regular price)."
    }
}


# ---------------------------------------------------------------------------
# TRANSACTIONS
# One or more per order. Covers completed, duplicate, partial refund,
# refund in progress, and wrong amount charged scenarios.
# ---------------------------------------------------------------------------

TRANSACTIONS = {

    "TXN-5001": {
        "transaction_id":   "TXN-5001",
        "order_id":         "ORD-1001",
        "customer_email":   "sarah.jones@email.com",
        "amount":           89.99,
        "payment_method":   "Visa ending 4242",
        "transaction_date": "2024-01-10",
        "status":           "completed",
        "refund_status":    None,
        "refund_amount":    None,
        "refund_date":      None
    },

    "TXN-5002": {
        "transaction_id":   "TXN-5002",
        "order_id":         "ORD-1002",
        "customer_email":   "mike.chen@email.com",
        "amount":           149.99,
        "payment_method":   "Mastercard ending 8888",
        "transaction_date": "2024-01-25",
        "status":           "completed",
        "refund_status":    None,
        "refund_amount":    None,
        "refund_date":      None
    },

    "TXN-5003": {
        # First of two charges for ORD-1003 (legitimate charge)
        "transaction_id":   "TXN-5003",
        "order_id":         "ORD-1003",
        "customer_email":   "priya.patel@email.com",
        "amount":           120.00,
        "payment_method":   "Visa ending 1111",
        "transaction_date": "2024-01-30",
        "status":           "completed",
        "refund_status":    None,
        "refund_amount":    None,
        "refund_date":      None
    },

    "TXN-5004": {
        # Duplicate charge for ORD-1003 — same day, same amount
        # Billing scenario: customer charged twice for one order
        "transaction_id":   "TXN-5004",
        "order_id":         "ORD-1003",
        "customer_email":   "priya.patel@email.com",
        "amount":           120.00,
        "payment_method":   "Visa ending 1111",
        "transaction_date": "2024-01-30",
        "status":           "completed",
        "refund_status":    None,
        "refund_amount":    None,
        "refund_date":      None,
        "notes":            "Duplicate charge. Should be refunded."
    },

    "TXN-5005": {
        # ORD-1004 defective coffee maker
        # Refund initiated 2024-01-24 after return approved, still processing after 8 days
        "transaction_id":   "TXN-5005",
        "order_id":         "ORD-1004",
        "customer_email":   "james.wilson@email.com",
        "amount":           75.00,
        "payment_method":   "Visa ending 5555",
        "transaction_date": "2024-01-05",
        "status":           "completed",
        "refund_status":    "processing",
        "refund_amount":    75.00,
        "refund_date":      None,
        "refund_initiated": "2024-01-24",
        "notes":            "Full refund initiated 2024-01-24 after return approval. "
                            "Not yet reflected in account as of 2024-02-01 (8 days)."
    },

    "TXN-5006": {
        # ORD-1005 cancelled order
        # Refund initiated 2024-01-21 after cancellation, still processing after 11 days
        "transaction_id":   "TXN-5006",
        "order_id":         "ORD-1005",
        "customer_email":   "lisa.martin@email.com",
        "amount":           55.00,
        "payment_method":   "Mastercard ending 3333",
        "transaction_date": "2024-01-20",
        "status":           "completed",
        "refund_status":    "processing",
        "refund_amount":    55.00,
        "refund_date":      None,
        "refund_initiated": "2024-01-21",
        "notes":            "Full refund initiated 2024-01-21 after cancellation. "
                            "Not yet reflected after 11 days. Exceeds standard 3-5 day window."
    },

    "TXN-5007": {
        # ORD-1006 high-value standing desk
        "transaction_id":   "TXN-5007",
        "order_id":         "ORD-1006",
        "customer_email":   "lisa.martin@email.com",
        "amount":           520.00,
        "payment_method":   "Visa ending 7777",
        "transaction_date": "2024-01-22",
        "status":           "completed",
        "refund_status":    None,
        "refund_amount":    None,
        "refund_date":      None
    },

    "TXN-5008": {
        # ORD-1007 laptop stand with partial refund dispute
        # $20 refund already completed, customer expects full $45
        "transaction_id":   "TXN-5008",
        "order_id":         "ORD-1007",
        "customer_email":   "sarah.jones@email.com",
        "amount":           45.00,
        "payment_method":   "Visa ending 4242",
        "transaction_date": "2024-01-10",
        "status":           "completed",
        "refund_status":    "partial",
        "refund_amount":    20.00,
        "refund_date":      "2024-01-20",
        "notes":            "Partial refund of $20.00 issued 2024-01-20. "
                            "Customer disputes: expects full $45.00 refund."
    },

    "TXN-5009": {
        # ORD-1008 bluetooth speaker, return window edge case
        "transaction_id":   "TXN-5009",
        "order_id":         "ORD-1008",
        "customer_email":   "mike.chen@email.com",
        "amount":           68.00,
        "payment_method":   "Mastercard ending 8888",
        "transaction_date": "2024-01-02",
        "status":           "completed",
        "refund_status":    None,
        "refund_amount":    None,
        "refund_date":      None
    },

    "TXN-5010": {
        # ORD-1009 kitchen blender, ambiguous delivery (Article 11)
        "transaction_id":   "TXN-5010",
        "order_id":         "ORD-1009",
        "customer_email":   "david.kim@email.com",
        "amount":           95.00,
        "payment_method":   "Visa ending 2222",
        "transaction_date": "2024-01-25",
        "status":           "completed",
        "refund_status":    None,
        "refund_amount":    None,
        "refund_date":      None
    },

    "TXN-5011": {
        # ORD-1010 smartwatch, wrong amount charged (Article 11)
        # Charged $89.99 instead of correct sale price $79.99
        "transaction_id":   "TXN-5011",
        "order_id":         "ORD-1010",
        "customer_email":   "david.kim@email.com",
        "amount":           89.99,           # charged amount (incorrect)
        "payment_method":   "Mastercard ending 9999",
        "transaction_date": "2024-01-28",
        "status":           "completed",
        "refund_status":    None,
        "refund_amount":    None,
        "refund_date":      None,
        "notes":            "Customer charged $89.99. Order placed at sale price $79.99. "
                            "Discrepancy: $10.00."
    }
}


# ---------------------------------------------------------------------------
# RETURN RECORDS
# One per order that has a return scenario.
# ---------------------------------------------------------------------------

RETURN_RECORDS = {

    "RET-001": {
        # ORD-1001 wrong item (jacket wrong size)
        # Return submitted but still under review after 14 days
        "return_id":             "RET-001",
        "order_id":              "ORD-1001",
        "customer_email":        "sarah.jones@email.com",
        "items":                 ["Blue Denim Jacket Size L (wrong size received)"],
        "reason":                "Wrong item received. Ordered Size M, received Size L.",
        "status":                "pending_review",
        "return_requested_date": "2024-01-18",
        "refund_amount":         89.99,
        "refund_status":         None,
        "notes":                 "Return submitted 2024-01-18. Under review 14 days. "
                                 "No update provided to customer."
    },

    "RET-002": {
        # ORD-1004 defective coffee maker
        # Return approved, full refund initiated but stuck processing for 8 days
        # Core scenario for James Wilson escalation
        "return_id":             "RET-002",
        "order_id":              "ORD-1004",
        "customer_email":        "james.wilson@email.com",
        "items":                 ["Coffee Maker BrewMaster 3000"],
        "reason":                "Defective. Unit does not power on.",
        "status":                "approved",
        "return_requested_date": "2024-01-14",
        "return_approved_date":  "2024-01-24",
        "refund_amount":         75.00,
        "refund_status":         "processing",
        "notes":                 "Return approved 2024-01-24. Full refund of $75.00 initiated "
                                 "same day. Not reflected in account as of 2024-02-01 (8 days). "
                                 "Exceeds standard 3-5 business day window."
    },

    "RET-003": {
        # ORD-1007 laptop stand partial refund dispute
        # Return closed with partial refund only. Customer expects full refund.
        "return_id":             "RET-003",
        "order_id":              "ORD-1007",
        "customer_email":        "sarah.jones@email.com",
        "items":                 ["Laptop Stand"],
        "reason":                "Item did not match description.",
        "status":                "closed",
        "return_requested_date": "2024-01-17",
        "return_approved_date":  "2024-01-19",
        "refund_amount":         20.00,       # partial refund only
        "refund_status":         "completed",
        "refund_date":           "2024-01-20",
        "notes":                 "Return closed with partial refund of $20.00. "
                                 "Full item price was $45.00. Customer disputes partial amount."
    }
}


# ---------------------------------------------------------------------------
# CONTACT HISTORY
# Drives escalation routing decisions.
# ---------------------------------------------------------------------------

CONTACT_HISTORY = {

    "sarah.jones@email.com": {
        "customer_email":  "sarah.jones@email.com",
        "contact_count":   1,
        "contacts": [
            {
                "date":    "2024-01-18",
                "issue":   "Reported wrong item received in ORD-1001. "
                           "Submitted return request RET-001.",
                "resolved": False
            }
        ],
        "unresolved_issues": ["Return RET-001 pending review for 14 days with no update."]
    },

    "mike.chen@email.com": {
        "customer_email":    "mike.chen@email.com",
        "contact_count":     0,
        "contacts":          [],
        "unresolved_issues": []
    },

    "priya.patel@email.com": {
        "customer_email":    "priya.patel@email.com",
        "contact_count":     0,
        "contacts":          [],
        "unresolved_issues": []
    },

    "james.wilson@email.com": {
        # Three prior contacts with the same unresolved refund issue.
        # Contact count exceeds escalation threshold (2). Strong escalation trigger.
        "customer_email":  "james.wilson@email.com",
        "contact_count":   3,
        "contacts": [
            {
                "date":    "2024-01-14",
                "issue":   "Reported defective coffee maker ORD-1004. "
                           "Requested return and refund.",
                "resolved": False
            },
            {
                "date":    "2024-01-19",
                "issue":   "Return approved but no refund update received.",
                "resolved": False
            },
            {
                "date":    "2024-01-28",
                "issue":   "Refund still not received. Customer expressed strong frustration.",
                "resolved": False
            }
        ],
        "unresolved_issues": [
            "Refund of $75.00 for ORD-1004 initiated 2024-01-24. "
            "Not received after 8 days. Three contacts with no resolution."
        ]
    },

    "lisa.martin@email.com": {
        "customer_email":  "lisa.martin@email.com",
        "contact_count":   1,
        "contacts": [
            {
                "date":    "2024-01-23",
                "issue":   "Asked about refund status for cancelled order ORD-1005.",
                "resolved": False
            }
        ],
        "unresolved_issues": [
            "Refund of $55.00 for cancelled ORD-1005 not received after 11 days."
        ]
    },

    "david.kim@email.com": {
        "customer_email":    "david.kim@email.com",
        "contact_count":     0,
        "contacts":          [],
        "unresolved_issues": []
    }
}