"""
Bounded clarification classifier (Article 9).

Given a piece of clarification text, classify it against the fixed
KNOWN_CLARIFICATION_KEYS, or signal "no_match" if it doesn't fit.

This is the only place an LLM touches clarification data on the write
path into ClarificationState. The output space is closed by construction
(see ClarificationKey in clarification_state.py) — the LLM is asked to
choose among a fixed set of options, never to invent a key. If it ever
does return something outside that set, Pydantic validation rejects it
and the code falls back to no_match, so the text is preserved in
unstructured_notes rather than entering the system under a key nothing
downstream knows how to read.

A pre-filter strips order IDs, emails, and phone numbers before the LLM
call. Those identifiers are already handled by regex in tools.py — the
classifier's job is only for facts that regex cannot capture. Passing
identifiers to the LLM risks misclassification regardless of how the
prompt describes each key.
"""

import re
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, ValidationError

from clarification_state import KNOWN_CLARIFICATION_KEYS, ClarificationKey

load_dotenv(find_dotenv())
client = OpenAI()


def _strip_identifiers(text: str) -> str:
    """
    Remove order IDs, email addresses, and phone numbers from
    clarification text before passing it to the classifier.

    Mirrors the extraction patterns in tools.py exactly, so what each
    specialist strips from a message and what the classifier strips
    before classification are always the same set of patterns.
    """
    # Order IDs: ORD-1001, order #1001, order 1001, #1001
    text = re.sub(
        r'\b(ORD-\d{3,6})\b|order\s*#?\s*(\d{3,6})\b|#(\d{3,6})\b',
        '', text, flags=re.IGNORECASE
    )
    # Email addresses
    text = re.sub(
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
        '', text
    )
    # Phone numbers: 617-555-0101, (617) 555-0101, 6175550101
    text = re.sub(r'\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}', '', text)
    return text.strip(" ,.\n")


class ClassificationResult(BaseModel):
    """
    Validated shape of the classifier's output. matched_key is
    constrained to ClarificationKey — an invented key fails validation
    here rather than silently entering ClarificationState under a name
    no specialist contract knows how to read.
    """
    matched_key: ClarificationKey
    extracted_value: str


def classify_clarification(clarification_text: str) -> ClassificationResult:
    """
    Classify a customer's clarification reply against the fixed key set.

    Returns a ClassificationResult with matched_key set to one of
    KNOWN_CLARIFICATION_KEYS, or "no_match" if nothing fits.
    On any parsing or validation failure, falls back to "no_match" with
    the original text preserved, so nothing is lost on a malformed
    classifier response.
    """
    # Strip identifiers before classification. If nothing meaningful
    # remains after stripping, return no_match immediately — no LLM
    # call needed for text that's purely identifying information.
    text_to_classify = _strip_identifiers(clarification_text)
    if not text_to_classify:
        print(f"[Clarification Classifier] matched_key: no_match "
              f"(identifiers only, no LLM call needed)")
        return ClassificationResult(
            matched_key="no_match",
            extracted_value=clarification_text
        )

    keys_str = " | ".join(KNOWN_CLARIFICATION_KEYS)

    key_descriptions = """Known categories and their precise meanings:

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
                           as identifying information."""

    system_prompt = f"""You are a classifier for customer clarification replies
in an e-commerce support pipeline.

You will be given a short piece of text the customer added in response to
a clarification request. Order IDs, email addresses, and phone numbers
have already been removed. Decide whether the remaining text matches
exactly one of these known fact categories:

{key_descriptions}

If the text clearly matches one category, return that key and extract
just the relevant portion as extracted_value.

If it does not clearly match any of these categories, return "no_match"
and set extracted_value to the original text unchanged.

Do not invent a new key. Only use one of the category names listed,
or "no_match".

Return ONLY a JSON object in this exact format:
{{
    "matched_key": "{keys_str} | no_match",
    "extracted_value": "the relevant text"
}}"""

    user_prompt = f"""Customer clarification reply (identifiers removed):
{text_to_classify}

Classify this reply."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )

    try:
        raw_result = json.loads(response.choices[0].message.content)
        result = ClassificationResult(**raw_result)
        print(f"[Clarification Classifier] matched_key: {result.matched_key}")
        return result

    except (json.JSONDecodeError, ValidationError) as e:
        # Closed-set validation failed — e.g. the LLM invented a key, or
        # returned malformed JSON. Fall back to no_match so the text is
        # preserved verbatim in unstructured_notes rather than dropped.
        print(f"[Clarification Classifier] Validation failed, "
              f"falling back to no_match: {e}")
        return ClassificationResult(
            matched_key="no_match",
            extracted_value=clarification_text
        )