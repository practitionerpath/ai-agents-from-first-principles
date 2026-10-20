# tools.py

import wikipedia

# ── Part 1: The actual function ──────────────────────────────────────────────

def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for a given query.
    Uses search to find the closest matching page — not exact title matching.
    Returns up to 1500 characters of the page summary.
    """
    try:
        # Step 1: search returns a list of matching page titles
        search_results = wikipedia.search(query, results=3)

        if not search_results:
            return f"No Wikipedia results found for: {query}"

        # Step 2: fetch the summary of the top result
        summary = wikipedia.summary(search_results[0], sentences=8)
        return summary[:1500]

    except wikipedia.exceptions.DisambiguationError as e:
        # The query matched a disambiguation page — try the first suggestion
        try:
            summary = wikipedia.summary(e.options[0], sentences=8)
            return summary[:1500]
        except Exception:
            return f"Ambiguous query. Closest options: {', '.join(e.options[:3])}"

    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for: {query}"

    except Exception as e:
        return f"Search failed: {str(e)}"


# ── Part 2: The schema OpenAI sees ───────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": (
                "Search Wikipedia for factual information about a topic. "
                "Use this when you need information to answer the user's question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic to search for on Wikipedia"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ── Part 3: The executor ──────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Routes a tool call from the agent loop to the correct function.
    """
    if tool_name == "search_wikipedia":
        return search_wikipedia(tool_args["query"])

    return f"Unknown tool: {tool_name}"


