import os
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import retriever
import verifier

load_dotenv(find_dotenv())

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LLM_MODEL = "gpt-4o-mini"
MAX_HISTORY_TURNS = 6   # 3 full exchanges (user + assistant each)

SYSTEM_PROMPT = """You are an expert SRE analyst specializing in production incident analysis. \
You have access to a curated knowledge base of real postmortem reports from Cloudflare and GitHub.

Rules you must follow:
1. Always call search_postmortems before answering. Never answer from your training knowledge alone.
2. Call search_postmortems multiple times if the question covers multiple aspects or your first results are insufficient.
3. Synthesize your answer only from what the retrieved chunks contain.
4. Attribute every specific claim to its source document by company and incident date.
5. If the indexed documents do not contain enough information to answer fully, say so explicitly \
rather than filling the gap from memory."""

# The retrieval function exposed as a tool.
# The LLM decides when to call it and what query to use.
# It may call it more than once per user turn.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_postmortems",
            "description": (
                "Search the indexed postmortem documents for relevant information. "
                "Use specific, focused queries for best results. "
                "You can call this multiple times with different queries to cover "
                "multiple aspects of a complex question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A specific search query. Prefer technical terms, "
                            "component names, or failure descriptions over general phrases."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def run_agent_turn(user_message: str, history: list[dict]) -> str:
    """
    Process one user turn through the full pipeline:
      agentic RAG loop --> draft answer --> verification --> final answer

    The agentic loop is the same pattern built in Article 2: call the LLM,
    execute any tool calls it requests, feed results back, repeat until it
    produces a text response with no tool calls.

    Verification is a second LLM call that checks whether the draft answer
    stays within what the retrieved chunks actually say. This catches the
    case where the agent draws on training data about well-known incidents
    rather than the indexed source documents.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_message}
    ]

    all_retrieved_chunks = []

    # --- Agentic RAG loop ---
    print()
    while True:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            # LLM wants to search. Execute each tool call and feed results back.
            # OpenAI can request multiple tool calls in one response (parallel
            # tool calling), so we iterate over all of them.
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                query = args["query"]
                print(f"  [search] \"{query}\"")

                chunks = retriever.search(query, top_k=5)
                all_retrieved_chunks.extend(chunks)

                # Tool results go back to the LLM as role="tool" messages.
                # The content is the formatted context block so the LLM sees
                # labeled source excerpts, not raw chunk text.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": retriever.format_context(chunks)
                })

        elif choice.finish_reason == "stop":
            # LLM decided it has enough context and produced its answer.
            draft_answer = choice.message.content
            break

        else:
            draft_answer = choice.message.content or "No answer generated."
            break

    # --- Verification ---
    # The draft answer might contain facts drawn from LLM training data
    # rather than from the retrieved chunks. The verifier checks each claim.
    print("  [verifying]")
    verification = verifier.verify(draft_answer, all_retrieved_chunks)

    if not verification["all_supported"] and verification["unsupported"]:
        print(f"  [first pass: {len(verification['unsupported'])} claim(s) need re-retrieval]")

        # One re-retrieval pass per unsupported claim.
        # Using the claim text as the query is more targeted than the
        # original user question and more likely to find supporting evidence
        # if it exists in the corpus.
        for claim in verification["unsupported"]:
            short = claim[:70] + "..." if len(claim) > 70 else claim
            print(f"  [re-retrieval] \"{short}\"")
            new_chunks = retriever.search(claim, top_k=3)
            all_retrieved_chunks.extend(new_chunks)

        # Re-verify with the expanded chunk set.
        # Claims still unsupported after this pass get flagged in the output.
        # We do not loop again — one re-retrieval attempt is the bound.
        print("  [re-verifying]")
        verification = verifier.verify(draft_answer, all_retrieved_chunks)

    return verifier.format_verified_answer(draft_answer, verification)


def main():
    print("=== Postmortem Analyzer ===")
    print("Ask questions across the indexed incidents. Type 'quit' to exit.\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            break

        final_answer = run_agent_turn(user_input, history)
        print(f"\nAgent: {final_answer}\n")

        # Store only the user question and the final answer in history.
        # The intermediate tool calls and retrieval results are ephemeral —
        # they exist for this turn's reasoning only and do not belong in the
        # ongoing conversation record.
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": final_answer})

        # Keep history to the last MAX_HISTORY_TURNS entries.
        if len(history) > MAX_HISTORY_TURNS:
            history = history[-MAX_HISTORY_TURNS:]


if __name__ == "__main__":
    main()