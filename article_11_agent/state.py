"""Graph state for the LangGraph rebuild of the retail triage chatbot.

Every node reads this state and returns only the fields it owns.
Ownership is noted per field. LangGraph does not enforce it; this file documents it.

TypedDict gives type hints, not runtime validation. Loud failures come from Pydantic
models at the boundaries that matter: classifier outputs and the assembled payload.
"""

import operator
from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Bounded vocabularies. An out-of-set value is a bug, not a possibility.
# ---------------------------------------------------------------------------

# What the customer wants. Specialist intents share their node's name.
Intent = Literal[
    "order_management",
    "returns_refunds",
    "billing_payments",
    "out_of_scope",
    "unclear",
]

# Outcome of the single, central order lookup.
ResolutionStatus = Literal["resolved", "not_found", "ambiguous", "missing"]

# Why a conversation needs a human. Every trigger except specialist_judgment
# is computed in code from tool data and STORE_POLICIES.
EscalationTrigger = Literal[
    "high_value",               # order total at or above policy threshold
    "repeat_contact",           # prior contacts at or above threshold, unresolved
    "delivered_unconfirmed",    # carrier and warehouse disagree
    "overdue_14_days",          # no delivery 14+ days past expected
    "return_window_closing",    # fewer than 3 days left
    "return_pending_10_days",   # return under review 10+ days
    "multiple_duplicates",      # more than one duplicate charge
    "refund_delay_10_days",     # refund processing 10+ days
    "wrong_amount_over_50",     # charge discrepancy above $50
    "specialist_judgment",      # LLM flagged genuine ambiguity; code chose to honor it
    "specialist_error",         # assessment failed; a human sees it instead of the customer
    "clarification_exhausted",  # attempt cap reached
]

ToneGuidance = Literal["standard", "empathetic", "priority", "de-escalation"]


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------

def merge_keyed(current: Optional[dict], update: Optional[dict]) -> dict:
    """Per-key merge: a new non-empty value replaces that key, missing keys are kept.

    The default reducer replaces the whole dict, so an email extracted on turn two
    would erase the order ID extracted on turn one. Used for identifiers and
    clarification facts: both are keyed snapshots of what is currently true.
    """
    merged = dict(current or {})
    for key, value in (update or {}).items():
        if value:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Structured field types
# ---------------------------------------------------------------------------

class Identifiers(TypedDict, total=False):
    """Customer-provided, regex-extracted from the latest message only."""
    order_id: str
    email: str
    phone: str


class Candidate(TypedDict):
    """An order offered to the customer for confirmation. Shown in the Clarify question."""
    order_id: str
    items: str
    order_date: str


class Resolution(TypedDict, total=False):
    """Result of the one lookup chain: order ID, then email, then phone.

    resolved_from records the identifiers this resolution was computed from.
    If current identifiers match, the lookup is skipped: a cache keyed by inputs.
    candidates are orders found by email or phone, offered to the customer when
    the named order was not found or several orders match.
    order_triggers are escalation triggers that depend only on the order and
    customer (high value, repeat contact), not on any specialist's analysis.
    """
    status: ResolutionStatus
    order_id: str
    customer_email: str
    order_triggers: list[EscalationTrigger]
    candidates: list[Candidate]
    resolved_from: Identifiers


class SpecialistResult(TypedDict):
    """System-of-record facts kept separate from LLM assessment.

    fetched_for stamps the resolved order ID this result belongs to. A mismatch
    with the current resolution means the result is stale and treated as absent.
    """
    fetched_for: str
    facts: dict                               # tool data plus code-computed analysis
    assessment: str                           # LLM
    recommended_action: str                   # LLM
    triggers: list[EscalationTrigger]         # code-computed from facts


class Escalation(TypedDict):
    triggers: list[EscalationTrigger]
    tone_guidance: ToneGuidance


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class ChatState(TypedDict, total=False):
    # Conversation. Owners: graph input, Clarify, Write response. Append-only.
    messages: Annotated[list, add_messages]

    # Customer-provided. Owner: Extract identifiers. Per-key merge.
    identifiers: Annotated[Identifiers, merge_keyed]

    # Central lookup. Owner: Resolve order. Overwrite.
    resolution: Resolution

    # Clarification facts (Article 9). Owner: Clarify.
    # facts: per-key snapshot. notes: verbatim text that matched no known key.
    # Article 9's facts_history is not carried over: checkpoint history records
    # every prior value of clarification_facts already.
    clarification_facts: Annotated[dict, merge_keyed]
    unstructured_notes: Annotated[list[str], operator.add]

    # Control. Owners: Supervisor (intents); clarification_attempts is a set/clear
    # pair: Clarify increments it, Write response resets it when a turn completes.
    # No hop counter: the only cycle (Clarify back through Supervisor) is bounded
    # by the clarification cap. LangGraph's recursion_limit guards against
    # routing bugs, not LLM behavior.
    intents: list[Intent]
    clarification_attempts: int

    # Workflow status, not data. Set True by Clarify when it resumes with the
    # customer's reply; set False by Supervisor once it has acted on it.
    # The one deliberate two-writer field: a set/clear pair, like a semaphore.
    awaiting_clarification: bool

    # System of record. One specialist owns each field. Overwrite.
    order_result: SpecialistResult
    return_result: SpecialistResult
    billing_result: SpecialistResult

    # Owner: Escalation.
    escalation: Escalation

    # Output. Owners: Assemble payload (payload), Write response (reply_text).
    # payload is validated by a Pydantic model in Assemble payload, stored as a dict.
    payload: dict
    reply_text: str