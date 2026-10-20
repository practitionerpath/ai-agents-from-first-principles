import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from tools import (
    extract_email,
    extract_phone,
    lookup_customer_by_email,
    lookup_customer_by_phone,
    get_contact_history,
    get_policy
)

load_dotenv(find_dotenv())
client = OpenAI()


def format_contact_history(history: dict | None) -> str:
    """
    Format contact history into a readable block for the LLM prompt.
    Returns a summary string whether or not history exists.
    """
    if not history:
        return "No contact history found for this customer."

    contact_count = history.get("contact_count", 0)
    if contact_count == 0:
        return "No prior contacts on record."

    contact_lines = []
    for c in history.get("contacts", []):
        resolved = "resolved" if c.get("resolved") else "unresolved"
        contact_lines.append(
            f"  {c['date']}: {c['issue']} ({resolved})"
        )

    unresolved = history.get("unresolved_issues", [])
    unresolved_str = (
        "\n".join([f"  - {u}" for u in unresolved])
        if unresolved else "  None"
    )

    return f"""Prior contact count: {contact_count}
Contact log:
{chr(10).join(contact_lines)}
Unresolved issues:
{unresolved_str}"""


def run_escalation(payload: dict) -> dict:
    """
    Escalation specialist.
    Receives: {customer_message, prior_context}
    Returns:  {assessment, recommended_action, needs_human,
               escalation_level, tone_guidance,
               needs_clarification, clarification_question}
    """
    customer_message = payload["customer_message"]
    prior_context    = payload.get("prior_context", "")

    # --- Step 1: Attempt customer lookup for contact history ---
    # Escalation does not look up orders. It looks up the customer
    # to access their prior contact history.
    email    = extract_email(customer_message)
    phone    = extract_phone(customer_message)
    customer = None
    history  = None

    if email:
        customer = lookup_customer_by_email(email)
        if customer:
            history = get_contact_history(email)

    if not customer and phone:
        customer = lookup_customer_by_phone(phone)
        if customer:
            history = get_contact_history(customer["email"])

    escalation_threshold = get_policy("escalation_contact_threshold")
    contact_count        = history.get("contact_count", 0) if history else 0
    has_unresolved       = bool(
        history.get("unresolved_issues") if history else False
    )
    threshold_exceeded   = contact_count >= escalation_threshold and has_unresolved

    print(f"[Escalation] contact_count: {contact_count}, "
          f"threshold_exceeded: {threshold_exceeded}, "
          f"has_unresolved: {has_unresolved}")

    # --- Step 2: Build context for LLM ---
    contact_history_str  = format_contact_history(history)
    customer_str = (
        f"{customer.get('name', 'Unknown')} "
        f"({customer.get('account_tier', 'standard').capitalize()} member)"
        if customer else "Customer not identified"
    )

    prior_section = (
        f"\nFindings from domain specialists already run:\n{prior_context}"
        if prior_context else ""
    )

    system_prompt = """You are an escalation specialist for an e-commerce customer service pipeline.

Your job is NOT to resolve the customer's order, billing, or returns issue.
That has already been handled by domain specialists.

Your job is to:
1. Assess the emotional tone and urgency of the customer's message
2. Review the contact history for repeat contacts and unresolved issues
3. Determine the escalation level and whether a human agent must be involved
4. Provide specific tone guidance so the writer produces the right kind of response

Escalation levels:
- low      : first contact, neutral tone, straightforward issue
- medium   : some frustration evident, or second contact with unresolved issue
- high     : clear frustration, multiple prior contacts, issue still unresolved,
             or explicit statement of dissatisfaction
- critical : legal threats, social media threats, VIP customer with major failure,
             or complete loss of trust expressed

Tone guidance options:
- standard      : professional and efficient, no special handling needed
- empathetic    : acknowledge frustration clearly before moving to resolution
- priority      : signal urgency, offer direct escalation path
- de-escalation : deeply empathetic, acknowledge repeated failures explicitly,
                  offer a meaningful concession or gesture before resolution steps

Return ONLY a JSON object in this exact format:
{
    "assessment": "summary of tone signals, contact history, and escalation triggers found",
    "recommended_action": "specific guidance for how the writer should handle this response",
    "needs_human": true | false,
    "escalation_level": "low" | "medium" | "high" | "critical",
    "tone_guidance": "standard" | "empathetic" | "priority" | "de-escalation",
    "needs_clarification": false,
    "clarification_question": ""
}

Set needs_human to true if:
- Contact count has reached or exceeded the escalation threshold with unresolved issues
- Escalation level is high or critical
- Customer explicitly threatens legal action or social media exposure
- Any domain specialist flagged needs_human in their assessment"""

    user_prompt = f"""Customer message:
{customer_message}

Customer profile:
  {customer_str}

Contact history:
{contact_history_str}

Escalation threshold: {escalation_threshold} prior contacts with unresolved issues
Threshold exceeded: {threshold_exceeded}{prior_section}

Assess the tone and urgency. Provide escalation level and tone guidance."""

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
        if "assessment" not in result or "escalation_level" not in result:
            raise ValueError("Missing required fields")

        result.setdefault("recommended_action",    "Follow tone guidance when drafting response.")
        result.setdefault("needs_human",           threshold_exceeded)
        result.setdefault("escalation_level",      "low")
        result.setdefault("tone_guidance",         "standard")
        result.setdefault("needs_clarification",   False)
        result.setdefault("clarification_question", "")

        # Code-level override: always flag when threshold is exceeded
        if threshold_exceeded:
            result["needs_human"] = True

        print(f"[Escalation] level: {result['escalation_level']}, "
              f"tone: {result['tone_guidance']}, "
              f"needs_human: {result['needs_human']}")
        return result

    except (json.JSONDecodeError, ValueError) as e:
        print(f"[Escalation] Failed to parse LLM response: {e}")
        return {
            "assessment":             "Unable to complete escalation assessment due to a processing error.",
            "recommended_action":     "Apply empathetic tone. Route to human agent.",
            "needs_human":            True,
            "escalation_level":       "high",
            "tone_guidance":          "empathetic",
            "needs_clarification":    False,
            "clarification_question": ""
        }