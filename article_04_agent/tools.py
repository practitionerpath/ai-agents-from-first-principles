import wikipedia
import requests


# ── Calculator ────────────────────────────────────────────────────────────────

def calculate(expression: str) -> str:
    try:
        allowed = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "pow": pow
        }
        result = eval(expression, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation failed: {str(e)}"


CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Calculate a mathematical expression. "
            "Use this when you need to do math to answer the user's question. "
            "Do not use this for encyclopedic information, history, or weather information."
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


# ── Weather ───────────────────────────────────────────────────────────────────

def search_weather(city_name: str) -> str:
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_name, "count": 1}
        ).json()

        results = geo.get("results")
        if not results:
            return f"City not found: {city_name}"

        lat = results[0]["latitude"]
        lon = results[0]["longitude"]

        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True}
        ).json()

        current = weather.get("current_weather")
        if not current:
            return f"No weather data returned for: {city_name}"

        return (
            f"Current weather in {city_name}: "
            f"Temperature: {current['temperature']}°C, "
            f"Wind speed: {current['windspeed']} km/h, "
            f"Weather code: {current['weathercode']}"
        )

    except Exception as e:
        return f"Weather lookup failed: {str(e)}"


WEATHER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_weather",
        "description": (
            "Get current weather for a city. "
            "Use this for real-time weather conditions only. "
            "Do not use this for encyclopedic information, history, or general knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city_name": {
                    "type": "string",
                    "description": "Name of the city to get weather for"
                }
            },
            "required": ["city_name"]
        }
    }
}


# ── Wikipedia ─────────────────────────────────────────────────────────────────

def search_wikipedia(query: str) -> str:
    try:
        search_results = wikipedia.search(query, results=3)

        if not search_results:
            return f"No Wikipedia results found for: {query}"

        summary = wikipedia.summary(search_results[0], sentences=8)
        return summary[:1500]

    except wikipedia.exceptions.DisambiguationError as e:
        try:
            summary = wikipedia.summary(e.options[0], sentences=8)
            return summary[:1500]
        except Exception:
            return f"Ambiguous query. Closest options: {', '.join(e.options[:3])}"

    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for: {query}"

    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"


WIKIPEDIA_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_wikipedia",
        "description": (
            "Search Wikipedia for factual, encyclopedic information — "
            "history, science, people, concepts. "
            "Do not use this for current conditions or real-time data like weather."
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


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS = [CALCULATOR_SCHEMA, WEATHER_SCHEMA, WIKIPEDIA_SCHEMA]

TOOL_FUNCTIONS = {
    "calculate": calculate,
    "search_weather": search_weather,
    "search_wikipedia": search_wikipedia
}