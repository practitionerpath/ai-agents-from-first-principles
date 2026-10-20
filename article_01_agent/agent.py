# agent.py

import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from tools import TOOLS, execute_tool

load_dotenv(find_dotenv())

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_ITERATIONS = 5  # hard ceiling — agent cannot loop forever


def run_agent(user_question: str) -> str:
    """
    The agent loop.
    Runs until the model returns a plain text answer (no tool call)
    or until MAX_ITERATIONS is hit.
    """

    # ── Conversation history ──────────────────────────────────────────────
    # This list IS the agent's memory for this task.
    # Every message — user, assistant, tool result — gets appended here.
    # The full list is sent to the API on every iteration.

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful research assistant. "
                "When you need factual information to answer a question, "
                "use the search_wikipedia tool. "
                "Once you have enough information, answer the user directly."
            )
        },
        {
            "role": "user",
            "content": user_question
        }
    ]

    # ── The loop ──────────────────────────────────────────────────────────

    for iteration in range(MAX_ITERATIONS):

        print(f"\n--- Iteration {iteration + 1} ---")

        # Step 1: Send full conversation history to the model
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"  # model decides whether to use a tool
        )

        # Step 2: Pull the response message out
        response_message = response.choices[0].message

        # Step 3: Check what came back — tool call or final answer?
        if response_message.tool_calls:

            # ── Tool call branch ──────────────────────────────────────────
            # The model wants to use a tool.
            # Append the assistant's decision to history first —
            # OpenAI requires the assistant message before the tool result.

            messages.append(response_message)

            # There can be multiple tool calls in one response.
            # Loop through all of them.
            for tool_call in response_message.tool_calls:

                tool_name = tool_call.function.name
                tool_args = eval(tool_call.function.arguments)

                print(f"Tool called: {tool_name}")
                print(f"Arguments:   {tool_args}")

                # Step 4: Execute the tool
                tool_result = execute_tool(tool_name, tool_args)

                print(f"Result preview: {tool_result[:200]}...")

                # Step 5: Append the tool result to history
                # Role must be "tool" — not "user" or "assistant"
                # tool_call_id links this result back to the specific call
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

            # Loop continues — model will reason over the tool result next

        else:

            # ── Final answer branch ───────────────────────────────────────
            # No tool call in the response — the model has its answer.
            # Extract the text and return it. Loop stops here.

            final_answer = response_message.content
            print(f"\nAgent finished in {iteration + 1} iteration(s).")
            return final_answer

    # ── Max iterations hit ────────────────────────────────────────────────
    # The loop ran MAX_ITERATIONS times without a final answer.
    # Return whatever the last message content was rather than failing silently.

    return "Max iterations reached without a final answer."