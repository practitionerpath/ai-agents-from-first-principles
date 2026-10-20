"""Bounded clarification classifier (carried over from Article 9).

Classifies free text in a clarification reply against a fixed key set, or
"no_match". Identifiers are stripped first: regex already handles those, and
the classifier's job is only the facts regex cannot capture.

Two changes from Article 9: the call goes through llm.py so it is counted, and
the key definitions live here since ClarificationState is replaced by state fields.
"""

import re
from typing import Literal

from pydantic import BaseModel

import llm

KNOWN_CLARIFICATION_KEYS = [
    "resolution_preference",   # e.g. refund vs. replacement vs. store credit
    "refund_reason",           # why the customer believes a refund is owed
    "delivery_instructions",   # e.g. leave with neighbour, alternate address
    "contact_preference",      # preferred contact method or time
]

ClarificationKey = Literal[
    "resolution_preference",
    "refund_reason",
    "delivery_instructions",
    "contact_preference",
    "no_match",
]

# A bare yes or no answers a confirmation question. It carries no fact to classify.
BARE_CONFIRMATION = re.compile(
    r"\s*(yes|yeah|yep|yup|no|nope|correct|right|that'?s (it|the one|right|correct))[\s.!]*",
    re.IGNORECASE,
)


def _strip_identifiers(text: str) -> str:
    """Remove order IDs, emails, and phone numbers. Mirrors the tools.py patterns."""
    text = re.sub(
        r'\b(ORD-\d{3,6})\b|order\s*#?\s*(\d{3,6})\b|#(\d{3,6})\b',
        '', text, flags=re.IGNORECASE,
    )
    text = re.sub(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b', '', text)
    text = re.sub(r'\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}', '', text)
    return text.strip(" ,.\n")


class ClassificationResult(BaseModel):
    matched_key: ClarificationKey
    extracted_value: str


_SYSTEM = f"""You are a classifier for customer clarification replies
in an e-commerce support pipeline.

You will be given a short piece of text the customer added in response to
a clarification request. Order IDs, email addresses, and phone numbers
have already been removed. Decide whether the remaining text matches
exactly one of these known fact categories:

  resolution_preference  : the customer explicitly states what outcome
                           they want, e.g. refund, replacement, exchange,
                           or store credit. Only use this when the customer
                           names a preferred resolution.

  refund_reason          : the customer explains why they believe a refund
                           is owed, e.g. item was defective, wrong item
                           received, item not as described. Only use this
                           when the customer gives a reason for a refund.

  delivery_instructions  : the customer provides instructions about where
                           or how to deliver a package, e.g. leave with
                           neighbour, deliver to alternate address. Only
                           use this when the customer gives delivery
                           guidance for a future or current shipment.

  contact_preference     : the customer explicitly states how or when they
                           want to be contacted, e.g. call me after 5pm,
                           please text rather than email. Do NOT use this
                           for an email address or phone number provided
                           as identifying information.

If the text clearly matches one category, return that key and extract
just the relevant portion as extracted_value.

If it does not clearly match any of these categories, return "no_match"
and set extracted_value to the original text unchanged.

Do not invent a new key. Only use one of the category names listed,
or "no_match".

Return ONLY a JSON object in this exact format:
{{
    "matched_key": "{' | '.join(KNOWN_CLARIFICATION_KEYS)} | no_match",
    "extracted_value": "the relevant text"
}}"""


def classify_clarification(reply: str) -> ClassificationResult | None:
    """Classify the non-identifier part of a reply.

    Returns None when there is nothing to classify: the reply was only
    identifiers, or only a yes/no confirmation. No LLM call in either case.
    """
    text = _strip_identifiers(reply)
    if not text or BARE_CONFIRMATION.fullmatch(text):
        print("[Clarify] nothing to classify beyond identifiers or yes/no, no LLM call")
        return None

    try:
        result = llm.structured(
            _SYSTEM,
            f"Customer clarification reply (identifiers removed):\n{text}\n\nClassify this reply.",
            ClassificationResult,
            label="clarification",
        )
    except ValueError as e:
        # Closed-set validation failed: preserve the text rather than drop it.
        print(f"[Clarify] classifier failed, keeping text as a note: {e}")
        return ClassificationResult(matched_key="no_match", extracted_value=text)

    print(f"[Clarify] classified: {result.matched_key}")
    return result