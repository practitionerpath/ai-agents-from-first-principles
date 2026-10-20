"""Runs every scenario non-interactively and checks each against its expectations.

One pass, as scoped for this article: a PASS shows the graph can take that path
with the real model, not that the model reliably takes it. A FAIL is recorded as
it happened, not rerun until it passes.

Usage:
    python3 scenarios.py            # summary only
    python3 scenarios.py --verbose  # also print every node's log line
"""

import contextlib
import io
import sys
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

import llm
from graph import build_graph
from main import TEST_MESSAGES

# ---------------------------------------------------------------------------
# Scenarios
# Each scenario is a list of turns. A turn is a customer message, the replies to
# any clarification questions in that turn, and what must be true afterwards.
#
# Expectation keys (all optional):
#   visits        nodes that must run during the turn
#   not_visits    nodes that must not run
#   kind          payload kind: answer, escalated, out_of_scope
#   triggers      escalation triggers that must be present
#   tone          exact tone guidance
#   reply_has     strings the reply must contain
#   fact          a clarification fact key that must be captured
# ---------------------------------------------------------------------------

def turn(message, replies=(), **expect):
    return {"message": message, "replies": list(replies), "expect": expect}


SCENARIOS = {
    # --- The Article 9 test messages, unchanged ---
    "T1 simple order status": [
        turn(TEST_MESSAGES["1"], visits=["order_management"],
             not_visits=["clarify", "escalation"], kind="answer", reply_has=["ORD-1002"]),
    ],
    "T2 wrong item, pending return": [
        turn(TEST_MESSAGES["2"], visits=["returns_refunds"],
             kind="escalated", triggers=["return_pending_10_days"]),
    ],
    "T3 duplicate charge": [
        turn(TEST_MESSAGES["3"], visits=["billing_payments"],
             not_visits=["escalation"], kind="answer", reply_has=["120"]),
    ],
    "T4 James: compound, repeat contact": [
        turn(TEST_MESSAGES["4"], visits=["returns_refunds", "billing_payments", "escalation"],
             kind="escalated", triggers=["repeat_contact"], tone="de-escalation",
             reply_has=["75"]),
    ],
    "T5 cancelled order, refund delayed": [
        turn(TEST_MESSAGES["5"], visits=["billing_payments"],
             kind="escalated", triggers=["refund_delay_10_days"], reply_has=["55"]),
    ],
    "T6 high value overdue": [
        turn(TEST_MESSAGES["6"], visits=["order_management"],
             kind="escalated", triggers=["high_value"]),
    ],
    "T7 return window, 5 days left": [
        turn(TEST_MESSAGES["7"], visits=["returns_refunds"],
             not_visits=["escalation"], kind="answer", reply_has=["ORD-1008"]),
    ],
    "T8 partial refund dispute": [
        turn(TEST_MESSAGES["8"], visits=["returns_refunds"], reply_has=["20", "45"]),
    ],
    "T9 vague, bad ID, then valid ID": [
        turn(TEST_MESSAGES["9"], replies=["ORD-9999", "ORD-1006 lisa.martin@email.com"],
             visits=["clarify", "order_management"], kind="escalated",
             triggers=["high_value"], reply_has=["ORD-1006"]),
    ],
    "T10 carrier says delivered, warehouse unconfirmed": [
        turn(TEST_MESSAGES["10"], visits=["order_management"],
             kind="escalated", triggers=["delivered_unconfirmed"]),
    ],
    "T11 vague, reply with preference": [
        turn(TEST_MESSAGES["11"],
             replies=["ORD-1010 and I'd actually rather get a replacement than a refund."],
             visits=["clarify"], fact="resolution_preference", reply_has=["ORD-1010"]),
    ],
    "T12 unknown order, delivery note, exhausted": [
        turn(TEST_MESSAGES["12"],
             replies=["ORD-1011, by the way the courier left it with my neighbor "
                      "without telling me.", "ORD-1011"],
             visits=["clarify", "escalation"], kind="escalated",
             triggers=["clarification_exhausted"], fact="delivery_instructions"),
    ],

    # --- New scenarios for the rebuild ---
    "N1 correction: order A, then order B": [
        turn("When will my order ORD-1002 arrive?",
             visits=["order_management"], reply_has=["ORD-1002"]),
        turn("Sorry, I meant ORD-1008.",
             visits=["order_management"], reply_has=["ORD-1008"]),
    ],
    "N2 order known, intent unclear": [
        turn("I have a question about ORD-1010.", replies=["It's about a charge."],
             visits=["clarify", "billing_payments"], reply_has=["ORD-1010"]),
    ],
    "N3 clarification exhausted": [
        turn(TEST_MESSAGES["9"], replies=["I don't know it.", "Sorry, I can't find it."],
             visits=["clarify", "escalation"], not_visits=["order_management"],
             kind="escalated", triggers=["clarification_exhausted"]),
    ],
    "N4 new question, same order, same thread": [
        turn("What's the status of my return for ORD-1007?", visits=["returns_refunds"]),
        turn("Also, was I charged correctly?",
             visits=["billing_payments"], not_visits=["returns_refunds", "clarify"]),
    ],
    "N5 customer claims a wrong amount": [
        turn("I was charged $99.99 for my smartwatch on ORD-1010. That's wrong.",
             visits=["billing_payments"], reply_has=["89.99"]),
    ],
    "N6 bad ID plus email, confirm the one candidate": [
        turn("I was charged twice. My order is ORD-9998 and my email is "
             "priya.patel@email.com.", replies=["yes"],
             visits=["clarify", "billing_payments"], reply_has=["ORD-1003"]),
    ],
    "N7 out of scope": [
        turn("What's the weather like in Boston today?",
             not_visits=["order_management", "returns_refunds", "billing_payments",
                         "clarify", "escalation"],
             kind="out_of_scope"),
    ],
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_turn(app, config, spec) -> tuple[list[str], list[str]]:
    """Run one turn, feeding scripted replies. Returns (nodes visited, problems)."""
    visited, problems = [], []
    replies = list(spec["replies"])

    def stream(payload):
        for chunk in app.stream(payload, config, stream_mode="updates"):
            visited.extend(k for k in chunk if k != "__interrupt__")

    stream({"messages": [HumanMessage(content=spec["message"])]})
    while app.get_state(config).interrupts:
        if not replies:
            question = app.get_state(config).interrupts[0].value["question"]
            problems.append(f"unexpected clarification: {question!r}")
            return visited, problems
        stream(Command(resume=replies.pop(0)))

    if replies:
        problems.append(f"{len(replies)} scripted reply(s) unused: expected clarification never came")
    return visited, problems


def check(state: dict, visited: list[str], expect: dict) -> list[str]:
    problems = []
    reply = state.get("reply_text")
    payload = state.get("payload", {})

    if not reply:
        return ["no reply produced"]

    for node in expect.get("visits", []):
        if node not in visited:
            problems.append(f"did not visit {node}")
    for node in expect.get("not_visits", []):
        if node in visited:
            problems.append(f"visited {node}, should not have")
    if "kind" in expect and payload.get("kind") != expect["kind"]:
        problems.append(f"kind {payload.get('kind')!r}, expected {expect['kind']!r}")
    for trigger in expect.get("triggers", []):
        if trigger not in payload.get("escalation_triggers", []):
            problems.append(f"missing trigger {trigger} "
                            f"(got {payload.get('escalation_triggers')})")
    if "tone" in expect and payload.get("tone_guidance") != expect["tone"]:
        problems.append(f"tone {payload.get('tone_guidance')!r}, expected {expect['tone']!r}")
    for text in expect.get("reply_has", []):
        if text not in reply:
            problems.append(f"reply missing {text!r}")
    if "fact" in expect and expect["fact"] not in state.get("clarification_facts", {}):
        problems.append(f"fact {expect['fact']!r} not captured "
                        f"(got {state.get('clarification_facts')})")
    return problems


def run_scenario(app, name, turns, verbose) -> bool:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    all_ok = True
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    for number, spec in enumerate(turns, 1):
        llm.reset_calls()
        log = io.StringIO()
        with contextlib.redirect_stdout(sys.stdout if verbose else log):
            try:
                visited, problems = run_turn(app, config, spec)
                state = app.get_state(config).values
                if not problems:
                    problems = check(state, visited, spec["expect"])
            except Exception as e:  # a crash is a FAIL, not an abort of the suite
                visited, state = [], {}
                problems = [f"crashed: {type(e).__name__}: {e}"]

        ok = not problems
        all_ok &= ok
        label = f"turn {number}: " if len(turns) > 1 else ""
        print(f"{'PASS' if ok else 'FAIL'}  {label}{spec['message'][:60]}")
        print(f"      path:  {' -> '.join(visited) or '(none)'}")
        print(f"      usage: {llm.call_summary()}")
        for problem in problems:
            print(f"      !! {problem}")
        if not ok and not verbose:
            print("      node log:")
            for line in log.getvalue().splitlines():
                print(f"        {line}")
        if state.get("reply_text"):
            print("      reply:")
            for line in state["reply_text"].splitlines():
                print(f"        {line}")
    return all_ok


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    app = build_graph()
    results = {name: run_scenario(app, name, turns, verbose)
               for name, turns in SCENARIOS.items()}

    passed = sum(results.values())
    print(f"\n{'=' * 70}\nSUMMARY: {passed}/{len(results)} passed\n{'=' * 70}")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")