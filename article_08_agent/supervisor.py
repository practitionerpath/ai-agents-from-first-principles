import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
client = OpenAI()


def format_specialist_outputs(state: dict) -> str:
    lines = []
    for name, output in state["specialist_outputs"].items():
        if output is None:
            lines.append(f"  {name}: not yet run")
        else:
            lines.append(f"  {name}:")
            lines.append(f"    assessment:          {output.get('assessment', 'N/A')}")
            lines.append(f"    recommended_action:  {output.get('recommended_action', 'N/A')}")
            lines.append(f"    needs_human:         {output.get('needs_human', False)}")
            lines.append(f"    needs_clarification: {output.get('needs_clarification', False)}")
            if output.get("needs_clarification"):
                lines.append(
                    f"    clarification_question: "
                    f"{output.get('clarification_question', 'N/A')}"
                )
    return "\n".join(lines)


def format_call_counts(state: dict) -> str:
    lines = []
    for name, count in state["specialist_call_counts"].items():
        status = f"called {count} time(s)" if count > 0 else "not yet called"
        lines.append(f"  {name}: {status}")
    return "\n".join(lines)


def format_routing_history(state: dict) -> str:
    if not state["routing_history"]:
        return "  none yet"
    return "\n".join([
        f"  step {i + 1}: routed to {h['next_agent']} — {h['reasoning']}"
        for i, h in enumerate(state["routing_history"])
    ])


def run_supervisor(state: dict) -> dict:
    """
    Supervisor agent.
    Receives: full pipeline state dict
    Returns:  {next_agent, payload, reasoning, escalation_required}
    """

    state_summary = f"""Customer message:
  {state['customer_message']}

Specialist call counts (max 2 calls per specialist):
{format_call_counts(state)}

Clarification attempts so far: {state['clarification_attempts']} (max 2)
Escalation flagged so far: {state['escalation_required']}

Specialist outputs:
{format_specialist_outputs(state)}

Routing history:
{format_routing_history(state)}"""

    system_prompt = """You are a supervisor agent for an e-commerce customer service pipeline.

Your job: read the customer message and current pipeline state, then decide what to do next.

Available agents and when to use them:
- order_management : delivery issues, wrong or missing items, tracking problems,
                     damaged goods on arrival
- returns_refunds  : return requests, exchange requests, refund eligibility,
                     return status, policy questions
- billing_payments : incorrect charges, duplicate billing, payment failures,
                     refund not received, charge disputes
- escalation       : frustrated tone, explicit mention of prior contacts,
                     threats of reviews or legal action.
                     Always run AFTER other relevant specialists. Never instead of them.
- clarify          : use when the customer message contains no identifying information
                     (no order ID, no email address, no phone number) and no specialist
                     has successfully looked up a record yet.
                     Also use when a specialist returned needs_clarification: true
                     AND clarification_attempts is below 2.
                     Always include a specific, friendly clarification_question in payload.
                     If clarification_attempts has reached 2 and no record has been found,
                     do NOT route to clarify again. Route to writer instead.
- writer           : drafts the final customer-facing response.
                     Call when all needed specialists have completed, OR when
                     clarification_attempts has reached 2 with no successful lookup.
- fail             : only if the message is entirely unrelated to e-commerce
                     or is completely incomprehensible.

Rules:
- A message may need more than one specialist. Route to one at a time.
- A specialist can be called at most twice. Do not exceed the call count limit.
- Only re-run a specialist if a subsequent specialist revealed information
  that materially changes the first specialist's assessment.
- Always include relevant prior specialist findings in prior_context so each
  specialist runs with full context. This reduces the need for re-runs.
- Escalation is a layer on top of domain specialists, not a replacement.
- When all needed specialists have run, route to writer.
- Always include the original customer message in every payload.

Return ONLY a JSON object in this exact format:
{
    "next_agent": "order_management" | "returns_refunds" | "billing_payments" |
                  "escalation" | "clarify" | "writer" | "fail",
    "payload": {
        "customer_message": "the original customer message",
        "prior_context": "summary of findings from specialists already run,
                          or empty string if none have run yet",
        "clarification_question": "only include when next_agent is clarify"
    },
    "reasoning": "one sentence explaining this routing decision",
    "escalation_required": true | false
}"""

    user_prompt = f"""Current pipeline state:

{state_summary}

What should happen next?"""

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

        if "next_agent" not in result or "payload" not in result:
            raise ValueError("Missing required fields in supervisor response")

        # Code-level guard: prevent routing to a specialist that has hit
        # its call limit regardless of what the LLM decides.
        next_agent = result["next_agent"]
        if next_agent in state["specialist_call_counts"]:
            if state["specialist_call_counts"][next_agent] >= 2:
                print(f"[Supervisor] {next_agent} has reached its call limit. "
                      f"Routing to writer instead.")
                result["next_agent"] = "writer"

        # Code-level guard: prevent routing to clarify when attempt
        # limit is reached regardless of what the LLM decides.
        if result["next_agent"] == "clarify" and state["clarification_attempts"] >= 2:
            print("[Supervisor] Clarification attempt limit reached. "
                  "Routing to writer for human handoff.")
            result["next_agent"] = "writer"

        # Writer payload is always constructed from state in code.
        # Do not trust the LLM to assemble specialist outputs correctly.
        # Also determine lookup_failed: true when clarification was exhausted
        # with no successful record lookups.
        if result["next_agent"] == "writer":
            useful_outputs = {
                k: v for k, v in state["specialist_outputs"].items()
                if v is not None and not v.get("needs_clarification", False)
            }
            lookup_failed = (
                state["clarification_attempts"] >= 2
                and len(useful_outputs) == 0
            )
            result["payload"] = {
                "customer_message":   state["customer_message"],
                "specialist_outputs": useful_outputs,
                "escalation_required": result.get(
                    "escalation_required", state["escalation_required"]
                ),
                "lookup_failed":      lookup_failed
            }

        return result

    except (json.JSONDecodeError, ValueError) as e:
        print(f"[Supervisor] Failed to parse response: {e}")
        return {
            "next_agent":          "fail",
            "payload":             {"reason": "Supervisor could not determine next action"},
            "reasoning":           "Parse error in supervisor response",
            "escalation_required": False
        }