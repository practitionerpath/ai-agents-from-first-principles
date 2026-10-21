from supervisor        import run_supervisor
from order_management  import run_order_management
from returns_refunds   import run_returns_refunds
from billing_payments  import run_billing_payments
from escalation        import run_escalation
from writer            import run_writer
from clarification_state      import ClarificationState
from clarification_classifier import classify_clarification

# Maximum pipeline steps before forcing a writer call.
# Prevents infinite loops if the supervisor keeps routing unexpectedly.
MAX_STEPS = 10

# Maps next_agent string to its runner function.
# All four specialists share the same call signature and output contract.
SPECIALIST_RUNNERS = {
    "order_management": run_order_management,
    "returns_refunds":  run_returns_refunds,
    "billing_payments": run_billing_payments,
    "escalation":       run_escalation
}


# ---------------------------------------------------------------------------
# TEST MESSAGES
# Use these to verify specific routing scenarios during the build.
# Run main.py and paste any of these when prompted for a customer message.
#
# Test 9 is reserved for the Article 8 before/after comparison: it is the
# exact message used in the original failing trace and must stay
# unmodified. At the clarify prompts, reproduce the original sequence:
#   first reply  -> ORD-9999
#   second reply -> ORD-1006 lisa.martin@email.com
#
# Tests 11 and 12 exercise the Article 9 clarification classifier and are
# new. Both deliberately omit identifying information so the supervisor
# routes to clarify, then the reply mixes an order id (handled by the
# existing regex path, unchanged) with text the classifier has to handle:
#   test 11 reply -> "ORD-1010 and I'd actually rather get a replacement
#                     than a refund." (matches resolution_preference)
#   test 12 reply -> "ORD-1011, by the way the courier left it with my
#                     neighbor without telling me." (matches no known key,
#                     falls through to unstructured_notes)
# ---------------------------------------------------------------------------

TEST_MESSAGES = {
    "1 - Simple order inquiry": (
        "Hi, I placed an order (ORD-1002) last week and wanted to check "
        "when it will arrive. Thanks, Mike"
    ),
    "2 - Wrong item received": (
        "I received the wrong size jacket in my order ORD-1001. "
        "I ordered size M but got size L. My email is sarah.jones@email.com. "
        "What do I do?"
    ),
    "3 - Duplicate charge": (
        "Hello, I was charged twice for my order ORD-1003. "
        "I can see two charges of $120 on my card from the same day. "
        "Please refund the extra charge. My email is priya.patel@email.com"
    ),
    "4 - Escalation: stuck refund": (
        "This is absolutely unacceptable. I have contacted you THREE times "
        "about my refund for order ORD-1004. The coffee maker was defective, "
        "my return was approved over a week ago and I still have not received "
        "my $75 refund. This is my last attempt before I dispute this with my bank. "
        "james.wilson@email.com"
    ),
    "5 - Cancelled order refund": (
        "I cancelled my order ORD-1005 almost two weeks ago and I still "
        "haven't received my refund of $55. "
        "My email is lisa.martin@email.com"
    ),
    "6 - High value overdue": (
        "My standing desk order ORD-1006 was supposed to arrive on January 28th "
        "and it's still not here. That was a $520 purchase. "
        "Please let me know what's happening. lisa.martin@email.com"
    ),
    "7 - Return window edge case": (
        "I'd like to return my bluetooth speaker from order ORD-1008. "
        "I know it's been a while but I think I'm still within the window. "
        "Email: mike.chen@email.com"
    ),
    "8 - Partial refund dispute": (
        "I returned my laptop stand from order ORD-1007 because it didn't "
        "match the description. I only got $20 back but I paid $45. "
        "Why wasn't I fully refunded? sarah.jones@email.com"
    ),
    "9 - No identifying info": (
        "My order hasn't arrived and I'm really frustrated. Can someone help me?"
    ),
    "10 - Ambiguous delivery": (
        "I was told my blender (ORD-1009) was delivered on January 30th "
        "but I never received it. There was nothing at my door. "
        "david.kim@email.com"
    ),
    "11 - Resolution preference clarification": (
        "Hi, I'm having trouble with a recent order and I'm not sure what "
        "to do. Can someone help me figure out the right next step?"
    ),
    "12 - Unstructured delivery note": (
        "Hello, I never got my package and I'm not sure who to contact "
        "about it. Can you help me look into this?"
    )
}


# ---------------------------------------------------------------------------
# PIPELINE HELPERS
# ---------------------------------------------------------------------------

def build_initial_state(customer_message: str) -> dict:
    return {
        "customer_message": customer_message,
        "clarification_state": ClarificationState(),
        "specialist_outputs": {
            "order_management": None,
            "returns_refunds":  None,
            "billing_payments": None,
            "escalation":       None
        },
        "specialist_call_counts": {
            "order_management": 0,
            "returns_refunds":  0,
            "billing_payments": 0,
            "escalation":       0
        },
        "clarification_attempts": 0,
        "escalation_required":    False,
        "routing_history":        [],
        "status":                 "triaging"
    }


def print_response(response: str) -> None:
    print("\n" + "=" * 60)
    print("CUSTOMER RESPONSE")
    print("=" * 60)
    print(response)
    print("=" * 60)


def print_specialist_summary(agent: str, result: dict) -> None:
    print(f"[{agent.replace('_', ' ').title()}] "
          f"needs_human: {result.get('needs_human', False)} | "
          f"needs_clarification: {result.get('needs_clarification', False)}")
    if result.get("needs_clarification"):
        print(f"  → {result.get('clarification_question', '')}")
    else:
        assessment = result.get("assessment", "")
        print(f"  → {assessment[:120]}{'...' if len(assessment) > 120 else ''}")


def print_routing_summary(state: dict) -> None:
    print("\n" + "-" * 60)
    print("ROUTING SUMMARY")
    print("-" * 60)
    for i, step in enumerate(state["routing_history"]):
        print(f"  Step {i + 1}: {step['next_agent']} — {step['reasoning']}")
    counts = {k: v for k, v in state["specialist_call_counts"].items() if v > 0}
    if counts:
        print(f"\nSpecialist calls: "
              f"{', '.join(f'{k}: {v}' for k, v in counts.items())}")
    print(f"Clarification attempts : {state['clarification_attempts']}")
    print(f"Final status           : {state['status']}")

    # Article 9: surface what the clarification classifier captured,
    # so a known-key match and an unstructured fallback are both visible
    # in the terminal output rather than only inferable from behavior.
    clarification_state = state.get("clarification_state")
    if clarification_state and (
        clarification_state.facts or clarification_state.unstructured_notes
    ):
        print("\nClarification facts gathered:")
        print(clarification_state.facts_summary())
        if clarification_state.unstructured_notes:
            print("Unstructured notes captured:")
            for note in clarification_state.unstructured_notes:
                print(f"  {note['value']}")

    print("-" * 60)


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(customer_message: str) -> None:
    state = build_initial_state(customer_message)
    print(f"\n[Pipeline] Starting. Message: "
          f"{customer_message[:80]}{'...' if len(customer_message) > 80 else ''}")

    for step in range(MAX_STEPS):
        print(f"\n{'─' * 40}")
        print(f"[Step {step + 1}] Calling supervisor...")

        decision   = run_supervisor(state)
        next_agent = decision.get("next_agent", "fail")
        payload    = decision.get("payload", {})
        reasoning  = decision.get("reasoning", "No reasoning provided")

        # Record this routing decision in history before acting on it
        state["routing_history"].append({
            "next_agent": next_agent,
            "reasoning":  reasoning
        })
        state["escalation_required"] = decision.get(
            "escalation_required", state["escalation_required"]
        )

        print(f"[Supervisor] → {next_agent}")
        print(f"[Supervisor] Reasoning: {reasoning}")

        # --- Specialist routing ---
        if next_agent in SPECIALIST_RUNNERS:
            result = SPECIALIST_RUNNERS[next_agent](payload)
            state["specialist_outputs"][next_agent]    = result
            state["specialist_call_counts"][next_agent] += 1
            state["status"] = "processing"
            print_specialist_summary(next_agent, result)

        # --- Clarification routing ---
        elif next_agent == "clarify":
            if state["clarification_attempts"] >= 2:
                print("[Main] Clarification limit reached. Routing to human handoff.")
                result = run_writer({
                    "customer_message":    state["customer_message"],
                    "specialist_outputs":  {},
                    "escalation_required": False,
                    "lookup_failed":       True
                })
                print_response(result["response"])
                state["status"] = "lookup_failed"
                break

            question = payload.get(
                "clarification_question",
                "Could you please provide your order number or "
                "the email address on your account?"
            )
            print(f"\n[Support] {question}")
            customer_input = input("Customer: ").strip()

            if customer_input:
                state["customer_message"] += (
                    f"\n[Customer follow-up]: {customer_input}"
                )

                # Article 9: classify the clarification text against the
                # known key schema. order_id/email/phone are still picked
                # up by each specialist's own regex on the full message
                # above — this classifier only handles facts regex cannot
                # capture, e.g. a stated resolution preference.
                classification = classify_clarification(customer_input)
                if classification.matched_key == "no_match":
                    state["clarification_state"].record_unstructured(
                        classification.extracted_value
                    )
                else:
                    state["clarification_state"].record_fact(
                        classification.matched_key,
                        classification.extracted_value
                    )

            state["clarification_attempts"] += 1
            state["status"] = "clarifying"

        # --- Writer routing ---
        elif next_agent == "writer":
            result = run_writer(payload)
            print_response(result["response"])
            state["status"] = "done"
            break

        # --- Failure routing ---
        elif next_agent == "fail":
            reason = payload.get("reason", "Unable to process request.")
            print(f"\n[Pipeline] Failed: {reason}")
            state["status"] = "failed"
            break

        else:
            print(f"\n[Pipeline] Unknown agent: {next_agent}. Stopping.")
            state["status"] = "failed"
            break

    else:
        # for/else: loop completed all MAX_STEPS without hitting a break.
        # Force writer with whatever has been collected.
        print(f"\n[Pipeline] Max steps ({MAX_STEPS}) reached. Forcing writer.")
        result = run_writer({
            "customer_message":    state["customer_message"],
            "specialist_outputs":  {
                k: v for k, v in state["specialist_outputs"].items()
                if v is not None
            },
            "escalation_required": state["escalation_required"],
            "lookup_failed":       False
        })
        print_response(result["response"])
        state["status"] = "max_steps_reached"

    print_routing_summary(state)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("E-COMMERCE CUSTOMER SERVICE TRIAGE SYSTEM")
    print("Article 09 — Agent Communication Patterns")
    print("=" * 60)

    print("\nTest messages available (paste the number to use one):")
    for key in TEST_MESSAGES:
        print(f"  {key}")

    while True:
        print("\nEnter a customer message (or 'quit' to exit):")
        raw_input = input("Customer: ").strip()

        if raw_input.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        if not raw_input:
            print("No message entered. Please try again.")
            continue

        # Allow test message selection by number prefix
        matched = next(
            (msg for key, msg in TEST_MESSAGES.items()
             if raw_input == key.split(" ")[0]),
            None
        )
        if matched:
            print(f"Using test message: {matched[:80]}...")
            run_pipeline(matched)
        else:
            run_pipeline(raw_input)

        print("\nProcess another message? (press Enter or 'quit' to exit)")
        again = input().strip().lower()
        if again in ("quit", "exit", "q"):
            print("Exiting.")
            break