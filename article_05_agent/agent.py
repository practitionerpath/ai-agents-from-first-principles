import json
from openai import OpenAI
from tools import TOOL_FUNCTIONS
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

TOOL_REGISTRY = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_context",
            "description": "Search the engineering knowledge base for relevant information about incidents, tickets, and technical issues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Use specific technical terms rather than the user's exact question."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are an engineering support assistant with access to a knowledge base of incident tickets.

When a question is asked, follow this process:
1. Think about what specific information you need to answer the question
2. Call retrieve_context with a focused search query to find relevant information
3. Read the retrieved content carefully
4. If the retrieved content references a specific ticket number, configuration parameter, or technical term that you have not yet looked up, call retrieve_context again with that specific term as your query
5. Once you have enough context from the knowledge base, provide a complete answer

Important: Do not use the user's exact question as your search query. Break it into what you need to find first, then follow what you learn."""


def run_agent(question):
    print(f"\nQuery: {question}")
    print("-" * 50)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    retrieval_count = 0

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOL_REGISTRY,
            tool_choice="auto"
        )

        message = response.choices[0].message
        messages.append(message)

        if response.choices[0].finish_reason == "tool_calls":
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                retrieval_count += 1
                print(f"\nRetrieval {retrieval_count}: '{tool_args['query']}'")

                result = TOOL_FUNCTIONS[tool_name](**tool_args)

                retrieved_sources = [
                    line.replace("[Source: ", "").replace("]", "")
                    for line in result.split("\n")
                    if line.startswith("[Source:")
                ]
                print(f"Returned: {retrieved_sources}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            print(f"\nAnswer:\n{message.content}")
            print(f"\nTotal retrievals: {retrieval_count}")
            return message.content


if __name__ == "__main__":
    print("=" * 60)
    print("QUERY 1: Single-hop")
    print("=" * 60)
    run_agent("What symptoms were users reporting during the authentication incident?")

    print("\n" + "=" * 60)
    print("QUERY 2: Multi-hop")
    print("=" * 60)
    run_agent("What caused the authentication failures and how was it fixed?")