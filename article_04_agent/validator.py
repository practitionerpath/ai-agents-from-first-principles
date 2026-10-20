import json
import re
from typing import Optional
from pydantic import BaseModel
from openai import OpenAI

from prompts import (
    build_confidence_prompt,
    build_grounding_check_prompt,
    build_correction_prompt
)


class ValidatedResponse(BaseModel):
    answer: str
    confidence_score: int
    grounded: bool
    tools_used: list[str]
    ungrounded_claims: list[str]
    warning: Optional[str] = None


class Validator:

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def validate(
        self,
        query: str,
        answer: str,
        session_tracker: dict
    ) -> ValidatedResponse:

        tools_used = list(session_tracker.keys())

        # Step 1: confidence elicitation
        confidence_score = self._get_confidence(query, answer, tools_used)

        # Step 2: tool-matched verification
        ungrounded_claims = []

        if "calculate" in session_tracker:
            issues = self._verify_calculator(
                answer, session_tracker["calculate"]
            )
            ungrounded_claims.extend(issues)

        if "search_weather" in session_tracker:
            issues = self._verify_weather(
                answer, session_tracker["search_weather"]
            )
            ungrounded_claims.extend(issues)

        if "search_wikipedia" in session_tracker:
            issues = self._verify_wikipedia(
                answer, session_tracker["search_wikipedia"]
            )
            ungrounded_claims.extend(issues)

        # Step 3: correction loop
        warning = None
        if ungrounded_claims:
            answer, still_ungrounded = self._correction_loop(
                answer, ungrounded_claims, session_tracker
            )
            if still_ungrounded:
                warning = (
                    f"Grounding check failed after revision. "
                    f"{len(still_ungrounded)} claim(s) could not be verified: "
                    + "; ".join(still_ungrounded[:2])
                )
                grounded = False
            else:
                grounded = True
        else:
            grounded = True

        return ValidatedResponse(
            answer=answer,
            confidence_score=confidence_score,
            grounded=grounded,
            tools_used=tools_used,
            ungrounded_claims=ungrounded_claims,
            warning=warning
        )

    def _get_confidence(
        self,
        query: str,
        answer: str,
        tools_used: list[str]
    ) -> int:
        tools_str = ", ".join(tools_used) if tools_used else "none"
        prompt = build_confidence_prompt(query, answer, tools_str)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            score = int(data.get("score", 5))
            return max(0, min(10, score))
        except Exception:
            return 5

    def _verify_calculator(
        self,
        answer: str,
        calc_calls: list[dict]
    ) -> list[str]:
        issues = []
        allowed = {"__builtins__": {}, "abs": abs, "round": round, "pow": pow}

        for call in calc_calls:
            expression = call.get("expression", "")
            stored_result = call.get("result", "")

            if not expression or not stored_result:
                continue

            # re-execute to verify the tool result was correct
            try:
                recalculated = str(eval(expression, allowed))
                if float(recalculated) != float(stored_result):
                    issues.append(
                        f"Calculation error: {expression} = {recalculated}, "
                        f"but tool returned {stored_result}"
                    )
                    continue
            except Exception:
                continue

            # check the answer states the correct result
            # handle both int and float representations: 360.0 and 360
            try:
                numeric = float(stored_result)
                int_form = str(int(numeric)) if numeric == int(numeric) else None
                stated_correctly = (
                    stored_result in answer or
                    (int_form is not None and int_form in answer)
                )
                if not stated_correctly:
                    issues.append(
                        f"Answer may not report the correct result for "
                        f"{expression} = {stored_result}"
                    )
            except ValueError:
                pass

        return issues

    def _verify_weather(
        self,
        answer: str,
        weather_calls: list[dict]
    ) -> list[str]:
        issues = []

        for call in weather_calls:
            city = call.get("city_name", "")
            raw = call.get("raw_result", "")

            if not raw:
                continue
            if raw.startswith("City not found") or raw.startswith("Weather lookup failed"):
                continue

            # extract temperature from the formatted tool result string
            temp_match = re.search(r"Temperature:\s*([\d.]+)°C", raw)
            if temp_match:
                temp_str = temp_match.group(1)
                try:
                    temp_val = float(temp_str)
                    int_temp = str(int(round(temp_val)))
                    stated_correctly = (
                        temp_str in answer or int_temp in answer
                    )
                    if not stated_correctly:
                        issues.append(
                            f"Answer may not accurately state the temperature "
                            f"for {city}: actual value was {temp_str}°C"
                        )
                except ValueError:
                    pass

        return issues

    def _verify_wikipedia(
        self,
        answer: str,
        wiki_calls: list[dict]
    ) -> list[str]:
        wiki_content = "\n\n".join(
            call.get("raw_result", "")
            for call in wiki_calls
            if call.get("raw_result")
        )

        if not wiki_content.strip():
            return []

        prompt = build_grounding_check_prompt(answer, wiki_content)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw = response.choices[0].message.content.strip()
            claims = json.loads(raw)
            return claims if isinstance(claims, list) else []
        except Exception:
            return []

    def _correction_loop(
        self,
        original_answer: str,
        ungrounded_claims: list[str],
        session_tracker: dict
    ) -> tuple[str, list[str]]:

        claims_text = "\n".join(f"- {c}" for c in ungrounded_claims)
        prompt = build_correction_prompt(original_answer, claims_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            revised = response.choices[0].message.content.strip()
        except Exception:
            return original_answer, ungrounded_claims

        # re-run wikipedia check on revised answer only
        # calculator and weather are deterministic — no need to re-check
        still_ungrounded = []
        if "search_wikipedia" in session_tracker:
            still_ungrounded = self._verify_wikipedia(
                revised, session_tracker["search_wikipedia"]
            )

        return revised, still_ungrounded