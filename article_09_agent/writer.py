import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
client = OpenAI()


def format_specialist_findings(specialist_outputs: dict) -> str:
    """
    Format specialist outputs into a clear summary for the writer LLM.
    Only includes specialists that ran and produced useful output.
    """
    lines = []

    order = specialist_outputs.get("order_management")
    if order:
        lines.append("ORDER MANAGEMENT FINDINGS:")
        lines.append(f"  Assessment       : {order.get('assessment', 'N/A')}")
        lines.append(f"  Recommended action: {order.get('recommended_action', 'N/A')}")
        lines.append(f"  Needs human      : {order.get('needs_human', False)}")
        lines.append("")

    returns = specialist_outputs.get("returns_refunds")
    if returns:
        lines.append("RETURNS AND REFUNDS FINDINGS:")
        lines.append(f"  Assessment       : {returns.get('assessment', 'N/A')}")
        lines.append(f"  Recommended action: {returns.get('recommended_action', 'N/A')}")
        lines.append(f"  Needs human      : {returns.get('needs_human', False)}")
        lines.append("")

    billing = specialist_outputs.get("billing_payments")
    if billing:
        lines.append("BILLING AND PAYMENTS FINDINGS:")
        lines.append(f"  Assessment       : {billing.get('assessment', 'N/A')}")
        lines.append(f"  Recommended action: {billing.get('recommended_action', 'N/A')}")
        lines.append(f"  Needs human      : {billing.get('needs_human', False)}")
        lines.append("")

    escalation = specialist_outputs.get("escalation")
    if escalation:
        lines.append("ESCALATION ASSESSMENT:")
        lines.append(f"  Assessment       : {escalation.get('assessment', 'N/A')}")
        lines.append(f"  Escalation level : {escalation.get('escalation_level', 'low')}")
        lines.append(f"  Tone guidance    : {escalation.get('tone_guidance', 'standard')}")
        lines.append(f"  Recommended action: {escalation.get('recommended_action', 'N/A')}")
        lines.append(f"  Needs human      : {escalation.get('needs_human', False)}")
        lines.append("")

    return "\n".join(lines) if lines else "No specialist findings available."


def any_needs_human(specialist_outputs: dict) -> bool:
    """
    Return True if any specialist flagged needs_human.
    """
    return any(
        output.get("needs_human", False)
        for output in specialist_outputs.values()
        if output is not None
    )


def get_tone_guidance(specialist_outputs: dict) -> tuple[str, str]:
    """
    Extract escalation level and tone guidance from escalation output.
    Returns (escalation_level, tone_guidance) with sensible defaults.
    """
    escalation = specialist_outputs.get("escalation")
    if escalation:
        return (
            escalation.get("escalation_level", "low"),
            escalation.get("tone_guidance", "standard")
        )
    return "low", "standard"


def run_writer(payload: dict) -> dict:
    """
    Response Writer.
    Receives: {customer_message, specialist_outputs,
               escalation_required, lookup_failed}
    Returns:  {response}
    """
    customer_message    = payload["customer_message"]
    specialist_outputs  = payload.get("specialist_outputs", {})
    escalation_required = payload.get("escalation_required", False)
    lookup_failed       = payload.get("lookup_failed", False)

    # --- Path 1: Lookup failed after two clarification attempts ---
    # Customer provided identifying information but no record was found.
    # Do not attempt to address the issue. Route to human agent.
    if lookup_failed:
        print("[Writer] Lookup failed path. Drafting human handoff response.")
        response_text = (
            "Thank you for reaching out. We were unable to locate an account "
            "or order matching the details you provided. This can sometimes happen "
            "if the order was placed under a different email address or phone number.\n\n"
            "A member of our support team will contact you within 24 hours to verify "
            "your details and resolve this personally. We apologise for the inconvenience "
            "and appreciate your patience."
        )
        return {"response": response_text}

    # --- Path 2: No specialist output available ---
    if not specialist_outputs:
        print("[Writer] No specialist outputs. Drafting generic acknowledgment.")
        response_text = (
            "Thank you for contacting us. We have received your message and a member "
            "of our support team will be in touch within 24 hours to assist you further."
        )
        return {"response": response_text}

    # --- Path 3: Standard response with specialist findings ---
    findings_str                    = format_specialist_findings(specialist_outputs)
    needs_human                     = any_needs_human(specialist_outputs)
    escalation_level, tone_guidance = get_tone_guidance(specialist_outputs)

    print(f"[Writer] Drafting response. tone: {tone_guidance}, "
          f"escalation_level: {escalation_level}, "
          f"needs_human: {needs_human}")

    tone_instructions = {
        "standard": (
            "Write in a professional and efficient tone. "
            "Be clear and direct. No need for extended empathy."
        ),
        "empathetic": (
            "Open with a genuine acknowledgment of the customer's frustration "
            "before moving to the resolution. Show that you understand the inconvenience."
        ),
        "priority": (
            "Signal clearly that this is being treated as a priority matter. "
            "Be direct about timelines and next steps. "
            "Do not bury the action items."
        ),
        "de-escalation": (
            "Open by acknowledging the repeated contacts and failures explicitly. "
            "Do not immediately jump to resolution. "
            "Validate the frustration first. "
            "Offer a meaningful gesture or concession before the resolution steps. "
            "Close with a direct commitment and a named timeline."
        )
    }.get(tone_guidance, "Write in a professional and helpful tone.")

    human_instruction = ""
    if needs_human:
        human_instruction = (
            "\nIMPORTANT: At least one issue requires human agent involvement. "
            "Include a clear statement that a member of the team will personally "
            "follow up within 24 hours. Name a specific commitment, not a vague promise."
        )

    system_prompt = f"""You are a customer service response writer for an e-commerce company.

Your job: draft a clear, helpful, customer-facing response based on the specialist 
findings provided. You are writing directly to the customer.

Tone instruction:
{tone_instructions}
{human_instruction}

Writing rules:
- Address every issue raised by the customer. Do not ignore any finding.
- Use the specialist recommended actions as your resolution steps.
- Do not use internal jargon, system IDs, or specialist names in the response.
- Order numbers are acceptable to include since customers know them.
- Do not make up information not present in the findings.
- Do not promise something a specialist did not recommend.
- Keep the response focused. Do not pad with generic pleasantries.
- If multiple issues are present, address each clearly with a brief heading
  or clear paragraph break so the customer can follow the resolution for each.
- Close with a clear next step or commitment.
- Always sign off with 'Customer Service Team'. Never use placeholder 
  names or bracketed variables like [Your Name].
  
Return ONLY a JSON object in this exact format:
{{
    "response": "the full customer-facing response text"
}}"""

    user_prompt = f"""Original customer message:
{customer_message}

Specialist findings:
{findings_str}

Escalation level : {escalation_level}
Needs human agent: {needs_human}

Draft the customer response now."""

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
        if "response" not in result:
            raise ValueError("Missing response field")

        print("[Writer] Response drafted successfully.")
        return result

    except (json.JSONDecodeError, ValueError) as e:
        print(f"[Writer] Failed to parse LLM response: {e}")
        return {
            "response": (
                "Thank you for contacting us. We have reviewed your message and "
                "a member of our support team will be in touch within 24 hours "
                "to resolve this for you personally. We apologise for any inconvenience."
            )
        }