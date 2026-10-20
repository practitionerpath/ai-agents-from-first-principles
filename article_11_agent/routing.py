"""Every routing decision in the graph.

Two decision points: after the Supervisor, and after any specialist.
Every other edge is fixed. No function here calls an LLM.

Escalation triggers are computed where their data lives (Resolve order and each
specialist). This module only aggregates them.
"""

from state import ChatState, EscalationTrigger

MAX_CLARIFICATION_ATTEMPTS = 2

# Specialist intent -> the state field holding its result.
# Intent names match node names, so an intent is also a routing target.
RESULT_FIELD = {
    "order_management": "order_result",
    "returns_refunds": "return_result",
    "billing_payments": "billing_result",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_fresh(state: ChatState, intent: str) -> bool:
    """A result counts only if it was fetched for the currently resolved order.

    A result stamped for order A while the resolution says order B is stale
    and treated as absent, so the specialist runs again.
    """
    result = state.get(RESULT_FIELD[intent])
    current_order = state.get("resolution", {}).get("order_id")
    return bool(result) and result["fetched_for"] == current_order


def next_specialist(state: ChatState) -> str | None:
    """First planned specialist without a fresh result, in plan order."""
    for intent in state.get("intents", []):
        if intent in RESULT_FIELD and not is_fresh(state, intent):
            return intent
    return None


def clarification_exhausted(state: ChatState) -> bool:
    """Cap reached AND still unable to proceed.

    Hitting the cap alone is not enough: a customer who supplies a valid order on
    their last allowed reply has been resolved, not exhausted.
    """
    intents = state.get("intents", [])
    status = state.get("resolution", {}).get("status")
    stuck = "unclear" in intents or status != "resolved"
    return stuck and state.get("clarification_attempts", 0) >= MAX_CLARIFICATION_ATTEMPTS


def collect_triggers(state: ChatState) -> list[EscalationTrigger]:
    """Order-level triggers plus triggers from fresh results of planned specialists.

    Stale results from earlier turns are excluded, so an old order's problems
    cannot escalate a conversation about a different order.
    """
    triggers = list(state.get("resolution", {}).get("order_triggers", []))
    for intent in state.get("intents", []):
        if intent in RESULT_FIELD and is_fresh(state, intent):
            triggers.extend(state[RESULT_FIELD[intent]]["triggers"])
    if clarification_exhausted(state):
        triggers.append("clarification_exhausted")
    return list(dict.fromkeys(triggers))  # de-duplicate, keep order


def _plan_complete(state: ChatState) -> str:
    """All planned specialists have fresh results: escalate once, or answer."""
    return "escalation" if collect_triggers(state) else "assemble_payload"


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def route_after_supervisor(state: ChatState) -> str:
    intents = state.get("intents", [])
    specialist_intents = [i for i in intents if i in RESULT_FIELD]

    # Nothing we handle: answer with a polite decline, no lookups.
    if not specialist_intents and "unclear" not in intents:
        return "assemble_payload"

    # Intent missing, or no single order to work on: ask, unless the cap is hit.
    if clarification_exhausted(state):
        return "escalation"
    status = state.get("resolution", {}).get("status")
    if "unclear" in intents or status != "resolved":
        return "clarify"

    return next_specialist(state) or _plan_complete(state)


def route_after_specialist(state: ChatState) -> str:
    return next_specialist(state) or _plan_complete(state)