# tools.py
import requests

# ── Part 1: The actual function ──────────────────────────────────────────────

def search_weather(city_name: str) -> str:
    """
    Search weather for a given query.
    """
    try:
        city_coordinates = requests.get("https://geocoding-api.open-meteo.com/v1/search?name=" + city_name + "&count=1").json()
        results = city_coordinates.get('results')
        if not results:
            return f"City not found: {city_name}"
        
        lat = city_coordinates['results'][0]['latitude']
        lon = city_coordinates['results'][0]['longitude']
        weather = requests.get("https://api.open-meteo.com/v1/forecast?latitude=" + str(lat) + "&longitude=" + str(lon) + "&current_weather=true").json()
        current = weather.get('current_weather')
        if not current:
            return f"No weather data returned for: {city_name}"
        return (
            f"Current weather in {city_name}: "
            f"Temperature: {current['temperature']}°C, "
            f"Wind speed: {current['windspeed']} km/h, "
            f"Weather code: {current['weathercode']}"
        )

    except Exception as e:
        return f"Search failed: {str(e)}"


# ── Part 2: The schema OpenAI sees ───────────────────────────────────────────

TOOL_SCHEMA = {
        "type": "function",
        "function": {
            "name": "search_weather",
            "description": (
                "Search weather for given city. "
                "Use this when you need current weather information for a city. "
                "Do not use this for encyclopedic information, history, or general knowledge. "
                "Use this when you need real-time weather data, not background information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "City Name to search weather for"
                    }
                },
                "required": ["city_name"]
            }
        }
    }
