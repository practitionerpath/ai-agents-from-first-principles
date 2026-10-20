from ddgs import DDGS


def search_web(query: str) -> str:
    results = DDGS().text(query, max_results=5)

    cleaned = []
    for r in results:
        title   = r.get("title", "")
        snippet = r.get("body", "")
        link    = r.get("href", "")
        cleaned.append(f"Source: {title}\n{snippet}\nURL: {link}")

    return "\n\n".join(cleaned) if cleaned else "No results found."


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the web for current information on a topic. "
            "Use this when you need facts, recent developments, "
            "or information not available from past research."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to run"
                }
            },
            "required": ["query"]
        }
    }
}