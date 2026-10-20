import json
import os
from openai import OpenAI
from memory import MemoryManager
from tools import search_web, SEARCH_TOOL
from prompts import SYSTEM_PROMPT, build_extraction_prompt, build_ranker_prompt


class ResearchAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.memory = MemoryManager()
        self.model  = "gpt-4o"

        self.TOOL_REGISTRY = {
            "search_web": search_web
        }
        self.ALL_TOOLS = [SEARCH_TOOL]

    def get_relevant_context(self, query: str) -> str | None:
        topics = self.memory.get_all_topics()
        if not topics:
            return None

        topic_list = "\n".join(
            f"ID {t['id']}: {t['topic']} ({t['timestamp'][:10]})"
            + (f" — {t['summary']}" if t.get('summary') else "")
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
            if ep['follow_ups']:
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

    def _run_agent_loop(self, messages: list) -> str:
        max_iterations = 5
        iterations = 0

        while True:
            if iterations >= max_iterations:
                print(f"[Agent] Max iterations reached — returning best available answer")
                for msg in reversed(messages):
                    if hasattr(msg, 'content') and msg.content:
                        return msg.content
                return "Research could not be completed within the allowed steps."

            iterations += 1

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.ALL_TOOLS,
                tool_choice="auto"
            )

            message = response.choices[0].message

            if not message.tool_calls:
                return message.content

            messages.append(message)

            for tool_call in message.tool_calls:
                name   = tool_call.function.name
                args   = json.loads(tool_call.function.arguments)
                result = (
                    self.TOOL_REGISTRY[name](**args)
                    if name in self.TOOL_REGISTRY
                    else f"Error: tool '{name}' not found in registry"
                )
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      result
                })

    def _extract_and_save(self, query: str, response: str) -> int | None:
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

        record_id = self.memory.save_episode(
            topic      = extracted.get("topic", "unknown"),
            query      = query,
            summary    = extracted.get("summary", ""),
            details    = extracted.get("details", ""),
            follow_ups = extracted.get("follow_ups")
        )

        print(f"[Memory] Saved episode #{record_id}: {extracted.get('topic')}")
        return record_id

    def run(self, query: str) -> str:
        print(f"\n[Agent] Processing: {query}")

        context = self.get_relevant_context(query)
        if context:
            print("[Memory] Found relevant past research")
        else:
            print("[Memory] No relevant past research — starting fresh")

        messages = self._build_messages(query, context)
        response = self._run_agent_loop(messages)
        self._extract_and_save(query, response)

        return response

    def close(self):
        self.memory.close()