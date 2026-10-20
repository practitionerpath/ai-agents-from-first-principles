"""Single gateway for every OpenAI call in the graph.

All LLM traffic goes through here so there is one place to count calls,
one model setting, and one way structured output is validated.
"""

import json
from collections import Counter
from typing import TypeVar

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(find_dotenv())
_client = OpenAI()

MODEL = "gpt-4o-mini"

# Calls per label for the current run. Module-level state: fine for a CLI and a
# scenario runner, wrong under concurrent requests.
_calls: Counter = Counter()

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Call counting
# ---------------------------------------------------------------------------

def reset_calls() -> None:
    _calls.clear()


def call_summary() -> str:
    """e.g. '5 calls: supervisor 1, returns_refunds 1, writer 1, ...'"""
    total = sum(_calls.values())
    detail = ", ".join(f"{label} {count}" for label, count in _calls.items())
    return f"{total} calls: {detail}" if total else "0 calls"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def transcript(messages: list) -> str:
    """Render LangGraph message objects as a plain transcript for a prompt."""
    speaker = {"human": "Customer", "ai": "Support"}
    return "\n".join(
        f"{speaker.get(m.type, m.type)}: {m.content}" for m in messages
    )


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------

def structured(system: str, user: str, schema: type[T], label: str) -> T:
    """JSON-mode call validated against a Pydantic schema.

    Raises ValueError on any bad response: empty content, invalid JSON
    (JSONDecodeError is a ValueError), or schema mismatch (ValidationError is a
    ValueError). Callers catch ValueError once. The calling node decides the
    fallback, so failure handling stays visible where the decision is made.

    Counts logical calls. SDK-internal retries are not counted separately.
    """
    _calls[label] += 1
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError(f"[{label}] empty response from model")
    return schema.model_validate(json.loads(content))


def text(system: str, user: str, label: str) -> str:
    """Free-text call. Used only for customer-facing prose.

    Raises ValueError on empty content, matching structured().
    """
    _calls[label] += 1
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError(f"[{label}] empty response from model")
    return content.strip()