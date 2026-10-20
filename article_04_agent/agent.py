from dotenv import load_dotenv
load_dotenv()

import json
import os
from openai import OpenAI
from memory import MemoryManager
from tools import TOOLS, TOOL_FUNCTIONS
from validator import Validator
from prompts import SYSTEM_PROMPT, build_extraction_prompt, build_ranker_prompt


class ResearchAgent:

    def __init__(self):
        self.client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.memory    = MemoryManager()
        self.validator = Validator(model="gpt-4o-mini")
        self.model     = "gpt-4o"

    def get_relevant_context(self, query: str) -> str | None:
        topics = self.memory.get_all_topics()
        if not topics:
            return None

        topic_list = "\n".join(
            f"ID {t['id']}: {t['topic']} ({t['timestamp'][:10]})"
            + (f" — {t.get('summary', '')}" if t.get("summary") else "")
            for t in topics
        )

        ranker_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": build_ranker_prompt(query, topic_list)
            }],
            temperature=0
        )

        raw = ranker_response.choices[0].message.content.strip()
        try:
            ids = json.loads(raw)
            if not isinstance(ids, list) or not ids:
                return None
        except json.JSONDecodeError:
            print(f"[Memory] Ranker returned unparseable response: {raw[:100]}")
            return None

        episodes = self.memory.get_episodes_by_ids(ids)
        if not episodes:
            return None

        context_parts = []
        for ep in episodes:
            part = (
                f"--- Past Research ({ep['timestamp'][:10]}) ---\n"
                f"Topic: {ep['topic']}\n"
                f"Summary: {ep['summary']}\n"
                f"Details: {ep['details']}\n"
            )
            if ep.get("follow_ups"):
                part += f"Open follow-ups: {ep['follow_ups']}\n"
            context_parts.append(part)

        return "\n".join(context_parts)

    def _build_messages(self, query: str, context: str | None) -> list:
        system_content = SYSTEM_PROMPT
        if context:
            system_content += f"\n\nRELEVANT PAST RESEARCH:\n{context}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": query}
        ]

    def _run_agent_loop(self, messages: list) -> tuple[str, dict]:
        max_iterations  = 5
        iterations      = 0
        session_tracker = {}

        while True:
            if iterations >= max_iterations:
                print("[Agent] Max iterations reached — returning best available answer")
                for msg in reversed(messages):
                    if hasattr(msg, "content") and msg.content:
                        return msg.content, session_tracker
                return "Research could not be completed within the allowed steps.", session_tracker

            iterations += 1

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )

            message = response.choices[0].message

            if not message.tool_calls:
                return message.content, session_tracker

            messages.append(message)

            for tool_call in message.tool_calls:
                name   = tool_call.function.name
                args   = json.loads(tool_call.function.arguments)
                result = (
                    TOOL_FUNCTIONS[name](**args)
                    if name in TOOL_FUNCTIONS
                    else f"Error: tool '{name}' not found"
                )

                # populate session tracker for validator
                if name not in session_tracker:
                    session_tracker[name] = []

                if name == "calculate":
                    session_tracker[name].append({
                        "expression": args.get("expression", ""),
                        "result":     result
                    })
                elif name == "search_weather":
                    session_tracker[name].append({
                        "city_name":  args.get("city_name", ""),
                        "raw_result": result
                    })
                elif name == "search_wikipedia":
                    session_tracker[name].append({
                        "query":      args.get("query", ""),
                        "raw_result": result
                    })

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      result
                })

    def _extract_and_save(
        self,
        query: str,
        response: str,
        confidence_score: int,
        source_reference: str
    ) -> int | None:

        extraction_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": build_extraction_prompt(query, response)
            }],
            temperature=0
        )

        raw = extraction_response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            extracted = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[Memory] Extraction failed — could not parse: {raw[:100]}")
            return None

        if not extracted.get("summary") and not extracted.get("details"):
            print("[Memory] Nothing meaningful to save")
            return None
        
        follow_ups = extracted.get("follow_ups")
        if isinstance(follow_ups, list):
            follow_ups = "; ".join(follow_ups)

        record_id = self.memory.save_episode(
            topic            = extracted.get("topic", "unknown"),
            query            = query,
            summary          = extracted.get("summary", ""),
            details          = extracted.get("details", ""),
            follow_ups       = follow_ups,
            confidence_score = confidence_score,
            source_reference = source_reference
        )

        print(f"[Memory] Saved episode #{record_id}: {extracted.get('topic')} "
              f"(confidence: {confidence_score}/10)")
        return record_id

    def run(self, query: str) -> str:
        print(f"\n[Agent] Processing: {query}")

        context = self.get_relevant_context(query)
        if context:
            print("[Memory] Found relevant past research")
        else:
            print("[Memory] No relevant past research — starting fresh")

        messages             = self._build_messages(query, context)
        raw_answer, session_tracker = self._run_agent_loop(messages)

        # validate before saving or returning
        validated = self.validator.validate(query, raw_answer, session_tracker)

        print(f"[Validator] Confidence: {validated.confidence_score}/10 | "
              f"Grounded: {validated.grounded} | "
              f"Tools used: {validated.tools_used or 'none'}")

        if validated.ungrounded_claims:
            print(f"[Validator] Ungrounded claims flagged: "
                  f"{len(validated.ungrounded_claims)}")

        if validated.warning:
            print(f"[Validator] Warning: {validated.warning}")

        self._extract_and_save(
            query            = query,
            response         = validated.answer,
            confidence_score = validated.confidence_score,
            source_reference = json.dumps(session_tracker)
        )

        return validated.answer

    def close(self):
        self.memory.close()