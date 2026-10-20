# executor.py

from tools.wikipedia_tool import search_wikipedia, TOOL_SCHEMA as WIKIPEDIA_SCHEMA
from tools.weather_tool import search_weather, TOOL_SCHEMA as WEATHER_SCHEMA
from tools.calculator_tool import calculate, TOOL_SCHEMA as CALCULATOR_SCHEMA

# The full tool list passed to the OpenAI API
ALL_TOOLS = [WIKIPEDIA_SCHEMA, WEATHER_SCHEMA, CALCULATOR_SCHEMA]


def execute_tool(tool_name: str, tool_args: dict) -> str:
    # your routing logic goes here
    if tool_name == "search_wikipedia":
        if tool_args.get("query") is None:
            return "Error: Missing required argument 'query' for search_wikipedia"
        return search_wikipedia(tool_args["query"])
    elif tool_name == "search_weather":
        if tool_args.get("city_name") is None:
            return "Error: Missing required argument 'city_name' for search_weather"
        return search_weather(tool_args["city_name"])
    elif tool_name == "calculate":
        if tool_args.get("expression") is None:
            return "Error: Missing required argument 'expression' for calculate"
        return calculate(tool_args["expression"])
    return f"Unknown tool: {tool_name}"