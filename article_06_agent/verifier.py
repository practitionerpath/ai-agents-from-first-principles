import os
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from retriever import format_context

load_dotenv(find_dotenv())

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LLM_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a fact-checker for an AI assistant that answers questions about production incident postmortems.

Your job is to check whether specific factual claims in a generated answer are explicitly supported by the retrieved source chunks provided.

Return ONLY a valid JSON object. No explanation, no preamble, no markdown code fences."""


def verify(draft_answer: str, retrieved_chunks: list[dict]) -> dict:
    """
    Check each factual claim in a draft answer against retrieved chunks.

    This is a second LLM call, separate from the agent that generated
    the draft. The separation matters: the agent optimizes for a good
    answer, the verifier optimizes for accuracy against the source.
    Both are LLM calls but they have different goals and different prompts.

    Returns a dict with three keys:
      claims        — list of {claim, status, source} for each claim found
      all_supported — True if every claim is SUPPORTED
      unsupported   — list of claim strings that need a re-retrieval pass
    """
    context = format_context(retrieved_chunks)

    user_prompt = f"""## Generated Answer
{draft_answer}

## Retrieved Source Chunks
{context}

## Instructions
Extract only specific factual claims from the generated answer. A factual claim is a
specific technical event, root cause, timeline detail, component name, duration, or
decision attributed to a named incident.

Do NOT extract:
- Analytical conclusions that summarise across incidents ("these examples demonstrate...")
- General observations about reliability or engineering practice
- Transition or framing sentences that connect ideas


For each factual claim, check whether it is directly and explicitly stated in one of
the source chunks above.

Return a JSON object in this exact format:
{{
  "claims": [
    {{
      "claim": "the exact factual claim text",
      "status": "SUPPORTED" or "UNSUPPORTED",
      "source": "Company | date | title if supported, otherwise null"
    }}
  ]
}}

Rules:
- SUPPORTED means a source chunk explicitly states this fact, not just implies it
- UNSUPPORTED means the fact cannot be found in any of the provided chunks
- General background definitions (e.g. "BGP is a routing protocol") are SUPPORTED with source null
- If the same fact appears in multiple chunks, use the most specific source
- Return only the JSON object, nothing else"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # verification must be deterministic, not creative
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model added them despite instructions.
    # This is a common failure mode even with explicit instructions not to.
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        # If the verifier fails to parse its own output, default to treating
        # everything as supported rather than crashing the agent loop.
        # The agent will surface an unverified answer which is better than
        # no answer, and the failure gets printed so you can debug it.
        print(f"  [verifier] JSON parse failed: {e}")
        print(f"  [verifier] Raw output: {raw[:300]}")
        return {
            "claims": [],
            "all_supported": True,
            "unsupported": []
        }

    claims = parsed.get("claims", [])
    unsupported = [
        c["claim"] for c in claims
        if c.get("status") == "UNSUPPORTED"
    ]

    return {
        "claims": claims,
        "all_supported": len(unsupported) == 0,
        "unsupported": unsupported
    }


def format_verified_answer(draft_answer: str, verification: dict) -> str:
    """
    Attach verification results to the draft answer.

    The LLM's own source attribution in the draft is left intact.
    This function only appends a warning section when claims could
    not be verified after the re-retrieval pass. Adding a second
    Sources block here caused duplication, so sources are omitted.
    """
    lines = [draft_answer.strip()]

    unverified = [
        c for c in verification["claims"]
        if c.get("status") == "UNSUPPORTED"
    ]

    if unverified:
        lines.append("")
        lines.append("Could not verify against indexed documents:")
        for c in unverified:
            lines.append(f"  - {c['claim']}")
        lines.append(
            "These claims may not be supported by the source postmortems."
        )

    return "\n".join(lines)