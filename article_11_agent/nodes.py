"""Node functions for the triage graph.

Each node reads ChatState and returns only the fields it owns.
Built in chunks: front of the graph first (code-only nodes).
"""

import json
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

import llm
from clarification_classifier import classify_clarification
from prompts import (
    RESPONSE_SYSTEM,
    SUPERVISOR_SYSTEM,
    TONE_SYSTEM,
    response_user,
    supervisor_user,
    tone_user,
)
from routing import RESULT_FIELD, collect_triggers, is_fresh
from state import (
    Candidate,
    ChatState,
    EscalationTrigger,
    Identifiers,
    Intent,
    Resolution,
    ToneGuidance,
)
from tools import (
    extract_email,
    extract_order_id,
    extract_phone,
    find_orders_by_email,
    find_orders_by_phone,
    get_contact_history,
    get_policy,
    lookup_customer_by_email,
    lookup_order_by_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latest_customer_message(state: ChatState) -> str:
    """Text of the most recent customer message. Earlier turns are never re-parsed."""
    for message in reversed(state.get("messages", [])):
        if message.type == "human":
            return message.content
    return ""


# ---------------------------------------------------------------------------
# 1. Extract identifiers (code only)
# ---------------------------------------------------------------------------

def extract_identifiers(state: ChatState) -> dict:
    """Regex-extract identifiers from the latest message only.

    Only non-empty values are returned; the merge_keyed reducer keeps earlier
    values for any key not mentioned this turn, and a new value replaces an old
    one. That is how "sorry, I meant ORD-1008" corrects an earlier order ID.
    """
    text = latest_customer_message(state)
    found: Identifiers = {
        key: value
        for key, value in {
            "order_id": extract_order_id(text),
            "email": extract_email(text),
            "phone": extract_phone(text),
        }.items()
        if value
    }
    print(f"[Extract identifiers] {found or 'none this turn'}")
    return {"identifiers": found}


# ---------------------------------------------------------------------------
# 2. Resolve order (code only)
# ---------------------------------------------------------------------------

# A short reply confirming a single offered order. Deterministic on purpose:
# a yes/no confirmation does not need an LLM.
AFFIRMATIVE = re.compile(
    r"^\s*(yes|yeah|yep|yup|correct|right|that'?s (it|the one|right|correct))\b",
    re.IGNORECASE,
)


def _order_triggers(order: dict) -> list[EscalationTrigger]:
    """Triggers that depend only on the order and customer, not on any specialist."""
    triggers: list[EscalationTrigger] = []

    if order["total"] >= get_policy("high_value_threshold"):
        triggers.append("high_value")

    history = get_contact_history(order["customer_email"]) or {}
    if (
        history.get("contact_count", 0) >= get_policy("escalation_contact_threshold")
        and history.get("unresolved_issues")
    ):
        triggers.append("repeat_contact")

    return triggers


def _candidate(order: dict) -> Candidate:
    return {
        "order_id": order["order_id"],
        "items": ", ".join(item["name"] for item in order["items"]),
        "order_date": order["order_date"],
    }


def _resolved(order: dict, ids: Identifiers) -> Resolution:
    return {
        "status": "resolved",
        "order_id": order["order_id"],
        "customer_email": order["customer_email"],
        "order_triggers": _order_triggers(order),
        "resolved_from": dict(ids),
    }


def _orders_by_contact(ids: Identifiers) -> list[dict]:
    if ids.get("email"):
        return find_orders_by_email(ids["email"])
    if ids.get("phone"):
        return find_orders_by_phone(ids["phone"])
    return []


def _confirmed_candidate(state: ChatState, ids: Identifiers) -> Resolution | None:
    """Customer said "yes" to the single order we offered last turn.

    Only honored while a clarification is pending, with exactly one candidate
    on offer, and a reply that is affirmative and names no new order ID.
    """
    previous = state.get("resolution", {})
    candidates = previous.get("candidates", [])
    if not state.get("awaiting_clarification") or len(candidates) != 1:
        return None

    reply = latest_customer_message(state)
    if extract_order_id(reply) or not AFFIRMATIVE.match(reply):
        return None

    order = lookup_order_by_id(candidates[0]["order_id"])
    return _resolved(order, ids) if order else None


def _lookup(ids: Identifiers) -> Resolution:
    """The lookup chain: order ID, then email, then phone.

    A named order ID that does not exist is never swapped for another order
    automatically. Orders found by email or phone are offered as candidates for
    the customer to confirm instead.
    """
    if ids.get("order_id"):
        order = lookup_order_by_id(ids["order_id"])
        if order:
            return _resolved(order, ids)
        return {
            "status": "not_found",
            "candidates": [_candidate(o) for o in _orders_by_contact(ids)],
            "resolved_from": dict(ids),
        }

    if not ids.get("email") and not ids.get("phone"):
        return {"status": "missing", "resolved_from": dict(ids)}

    orders = _orders_by_contact(ids)
    if len(orders) == 1:
        return _resolved(orders[0], ids)
    return {
        "status": "ambiguous" if orders else "not_found",
        "candidates": [_candidate(o) for o in orders],
        "resolved_from": dict(ids),
    }


def resolve_order(state: ChatState) -> dict:
    """Resolve the conversation to one order, or explain why it cannot.

    Check order matters:
      1. Confirmation: a "yes" turn has unchanged identifiers by definition,
         so it must be checked before the skip.
      2. Skip: identifiers unchanged since the last resolution means no lookup.
         Returning no update leaves the previous resolution in place.
      3. Lookup.
    """
    ids: Identifiers = state.get("identifiers", {})

    confirmed = _confirmed_candidate(state, ids)
    if confirmed:
        print(f"[Resolve order] confirmed by customer -> {confirmed['order_id']}")
        return {"resolution": confirmed}

    previous = state.get("resolution")
    if previous and previous.get("resolved_from") == ids:
        print(f"[Resolve order] identifiers unchanged, lookup skipped "
              f"({previous['status']})")
        return {}

    resolution = _lookup(ids)
    print(
        f"[Resolve order] {resolution['status']}"
        + (f" -> {resolution['order_id']}" if resolution.get("order_id") else "")
        + (f" | candidates: {[c['order_id'] for c in resolution['candidates']]}"
           if resolution.get("candidates") else "")
        + (f" | triggers: {resolution['order_triggers']}"
           if resolution.get("order_triggers") else "")
    )
    return {"resolution": resolution}


# ---------------------------------------------------------------------------
# 3. Supervisor (LLM only when needed)
# ---------------------------------------------------------------------------

class IntentPlan(BaseModel):
    """Classifier output. Literal-typed: an invented intent fails validation."""
    intents: list[Intent] = Field(min_length=1)
    reasoning: str


def normalize_intents(intents: list[Intent]) -> list[Intent]:
    """Enforce the rule the prompt only requests.

    "unclear" and "out_of_scope" are valid only alone. If any specialist intent
    is present, keep only specialist intents, in a fixed order, de-duplicated.
    """
    specialists = [i for i in RESULT_FIELD if i in intents]
    if specialists:
        return specialists
    if "unclear" in intents:
        return ["unclear"]
    return ["out_of_scope"]


def supervisor(state: ChatState) -> dict:
    """Decide what the customer wants. Routing decides what happens next.

    Pre-check: if we paused for clarification and already knew the intent,
    resume it without an LLM call. Whether the reply supplied enough data is
    routing's question, not this node's.
    """
    prior = state.get("intents", [])
    if state.get("awaiting_clarification") and any(i in RESULT_FIELD for i in prior):
        print(f"[Supervisor] resuming {prior} after clarification, no LLM call")
        return {"awaiting_clarification": False}

    messages = state.get("messages", [])
    try:
        plan = llm.structured(
            SUPERVISOR_SYSTEM,
            supervisor_user(llm.transcript(messages[:-1]), latest_customer_message(state)),
            IntentPlan,
            label="supervisor",
        )
        intents = normalize_intents(plan.intents)
        print(f"[Supervisor] {intents} | {plan.reasoning}")
    except ValueError as e:
        # A failed classification is treated as not knowing the intent.
        intents = ["unclear"]
        print(f"[Supervisor] classification failed, treating as unclear: {e}")

    return {"intents": intents, "awaiting_clarification": False}


# ---------------------------------------------------------------------------
# 7. Clarify (template question, pause, classify the reply)
# ---------------------------------------------------------------------------

ASK_INTENT = "Is your question about delivery, a return, or a charge?"
ASK_ORDER = ("Could you share your order number (it looks like ORD-1234) "
             "or the email address on your account?")


def _describe(candidate: Candidate) -> str:
    return f"{candidate['order_id']}: {candidate['items']} (ordered {candidate['order_date']})"


def clarification_question(state: ChatState) -> str:
    """Build the question from templates. Deterministic, so it is safe to run twice.

    Every template asks only for what Extract identifiers can parse, or for a
    yes/no that Resolve order can match. The question and its parsers cannot drift.
    """
    resolution = state.get("resolution", {})
    status = resolution.get("status")
    candidates = resolution.get("candidates", [])
    unclear = "unclear" in state.get("intents", [])
    named = state.get("identifiers", {}).get("order_id")
    retry = "Sorry, I still couldn't match that. " if state.get("clarification_attempts", 0) else ""

    if status == "resolved":
        # Only reason to be here with a resolved order: the intent is unclear.
        return f"{retry}I've found order {resolution['order_id']}. {ASK_INTENT}"

    if len(candidates) == 1:
        lead = f"I couldn't find order {named}, but " if named else ""
        question = (f"{retry}{lead}I see one order on your account: "
                    f"{_describe(candidates[0])}. Is that the one? (yes/no)")
    elif candidates:
        lead = f"I couldn't find order {named}. " if named else ""
        listing = "\n".join(f"  - {_describe(c)}" for c in candidates)
        question = (f"{retry}{lead}I see these orders on your account:\n{listing}\n"
                    f"Which one is this about? Please reply with the order number.")
    elif status == "not_found":
        question = (f"{retry}I couldn't find an order matching "
                    f"{named or 'those details'}. {ASK_ORDER}")
    else:  # missing
        question = f"{retry}{ASK_ORDER}"

    # Both the order and the intent are unknown: ask for both at once,
    # so one reply can resolve both and only one attempt is spent.
    return f"{question} {ASK_INTENT}" if unclear else question


def clarify(state: ChatState) -> dict:
    """Ask, pause, and record the customer's reply.

    On resume this node re-runs from the top. Everything before interrupt() is
    deterministic (a template), so re-running it recomputes the same question and
    costs nothing. The classifier call comes after interrupt(), so it runs once.
    """
    question = clarification_question(state)
    reply = interrupt({"question": question})

    update = {
        "messages": [AIMessage(content=question), HumanMessage(content=reply)],
        "clarification_attempts": state.get("clarification_attempts", 0) + 1,
        "awaiting_clarification": True,
    }

    result = classify_clarification(reply)
    if result and result.matched_key != "no_match":
        update["clarification_facts"] = {result.matched_key: result.extracted_value}
    elif result:
        update["unstructured_notes"] = [result.extracted_value]

    return update


# ---------------------------------------------------------------------------
# 8. Escalation (tone floor in code, emotion from the LLM)
# ---------------------------------------------------------------------------

TONE_ORDER: list[ToneGuidance] = ["standard", "empathetic", "priority", "de-escalation"]

# Facts that imply a minimum tone regardless of how calm the customer sounds.
TONE_FLOORS: dict[str, ToneGuidance] = {
    "repeat_contact": "de-escalation",           # we have failed them before
    "clarification_exhausted": "empathetic",     # three rounds without being understood
}


def _higher(a: ToneGuidance, b: ToneGuidance) -> ToneGuidance:
    return max(a, b, key=TONE_ORDER.index)


class ToneAssessment(BaseModel):
    tone_guidance: ToneGuidance
    reason: str


def escalation(state: ChatState) -> dict:
    """Record every trigger, and set the tone for the reply.

    Runs once, after all planned specialists, so the handoff carries every
    trigger rather than only the first one found.
    """
    triggers = collect_triggers(state)
    floor: ToneGuidance = "standard"
    for trigger in triggers:
        floor = _higher(floor, TONE_FLOORS.get(trigger, "standard"))

    try:
        out = llm.structured(
            TONE_SYSTEM,
            tone_user(llm.transcript(state.get("messages", []))),
            ToneAssessment,
            label="escalation",
        )
        tone = _higher(out.tone_guidance, floor)
        print(f"[Escalation] triggers: {triggers} | LLM tone: {out.tone_guidance}, "
              f"floor: {floor} -> {tone}")
    except ValueError as e:
        tone = _higher("empathetic", floor)
        print(f"[Escalation] tone call failed, using {tone}: {e}")

    return {"escalation": {"triggers": triggers, "tone_guidance": tone}}


# ---------------------------------------------------------------------------
# 9. Assemble payload (code only, validated)
# ---------------------------------------------------------------------------

class ResultSummary(BaseModel):
    specialist: str
    order_id: str
    facts: dict
    assessment: str
    recommended_action: str


class ResponsePayload(BaseModel):
    """The contract between state and the Response Writer.

    Validated here so a malformed payload fails loudly in code, before any
    LLM writes prose from it. State is a TypedDict and is never checked at runtime.
    """
    kind: Literal["answer", "escalated", "out_of_scope"]
    customer_name: str | None = None
    order_id: str | None = None
    results: list[ResultSummary] = []
    escalation_triggers: list[EscalationTrigger] = []
    tone_guidance: ToneGuidance = "standard"
    clarification_facts: dict = {}
    unstructured_notes: list[str] = []


def assemble_payload(state: ChatState) -> dict:
    """Copy facts from state into the payload. Nothing here is generated.

    Only fresh results for this turn's planned specialists are included, so a
    stale result for a previous order can never reach the reply.
    """
    intents = state.get("intents", [])
    resolution = state.get("resolution", {})

    if "out_of_scope" in intents:
        kind, triggers, tone = "out_of_scope", [], "standard"
    elif collect_triggers(state):
        # Routing escalates exactly when triggers exist, so Escalation ran this
        # turn and its tone is current.
        escalation_record = state["escalation"]
        kind = "escalated"
        triggers, tone = escalation_record["triggers"], escalation_record["tone_guidance"]
    else:
        kind, triggers, tone = "answer", [], "standard"

    results = [
        ResultSummary(specialist=intent, order_id=state[RESULT_FIELD[intent]]["fetched_for"],
                      **{k: state[RESULT_FIELD[intent]][k]
                         for k in ("facts", "assessment", "recommended_action")})
        for intent in intents
        if intent in RESULT_FIELD and is_fresh(state, intent)
    ]

    customer = (lookup_customer_by_email(resolution["customer_email"])
                if resolution.get("customer_email") else None)

    payload = ResponsePayload(
        kind=kind,
        customer_name=customer["name"] if customer else None,
        order_id=resolution.get("order_id"),
        results=results,
        escalation_triggers=triggers,
        tone_guidance=tone,
        clarification_facts=state.get("clarification_facts", {}),
        unstructured_notes=state.get("unstructured_notes", []),
    )
    print(f"[Assemble payload] kind: {kind} | results: "
          f"{[r.specialist for r in results] or 'none'} | tone: {tone}")
    return {"payload": payload.model_dump()}


# ---------------------------------------------------------------------------
# 10. Write response (LLM, prose only)
# ---------------------------------------------------------------------------

FALLBACK_REPLY = ("Thanks for your patience. I wasn't able to put together a full answer "
                  "just now, so a member of our support team will follow up with you "
                  "personally.\n\nCustomer Support")


def write_response(state: ChatState) -> dict:
    payload = state["payload"]
    try:
        reply = llm.text(
            RESPONSE_SYSTEM,
            response_user(llm.transcript(state.get("messages", [])),
                          json.dumps(payload, indent=2, default=str)),
            label="writer",
        )
    except ValueError as e:
        print(f"[Write response] failed, using fallback: {e}")
        reply = FALLBACK_REPLY

    # The prompt asks for the order number; code guarantees it. A reply about
    # the wrong order is then visible to the customer and correctable, not silent.
    order_id = payload.get("order_id")
    if order_id and order_id not in reply:
        reply = f"Regarding order {order_id}:\n\n{reply}"

    # A reply ends the clarification episode. Resetting here, where every
    # completed turn ends, needs no condition: a new request gets fresh attempts.
    return {
        "reply_text": reply,
        "messages": [AIMessage(content=reply)],
        "clarification_attempts": 0,
    }