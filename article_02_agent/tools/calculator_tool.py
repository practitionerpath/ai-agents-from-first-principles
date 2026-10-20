# tools.py
# ── Part 1: The actual function ──────────────────────────────────────────────

def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.
    """
    try:
        allowed = {
        "__builtins__": {},   # removes all built-ins — no imports, no os, nothing
        "abs": abs,
        "round": round,
        "pow": pow
        }
        result = eval(expression, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation failed: {str(e)}"


# ── Part 2: The schema OpenAI sees ───────────────────────────────────────────

TOOL_SCHEMA = {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Calculate a mathematical expression. "
                "Use this when you need to do math to answer the user's question. "
                "Do not use this for encyclopedic information, history, or weather information search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to calculate, e.g. (2 + 2) * 3"
                    }
                },
                "required": ["expression"]
            }
        }
    }
