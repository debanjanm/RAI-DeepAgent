import json
import datetime
import requests
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from rich.console import Console
from rich.theme import Theme

# Setup
console = Console(theme=Theme({"info": "cyan", "error": "bold red", "success": "green"}))
set_tracing_disabled(True)

# Models
class GetWeatherInput(BaseModel):
    location: str = Field(..., description="City and optionally state/country (e.g., 'London, UK')")

class WeatherInfo(BaseModel):
    location: str
    temperature_celsius: float
    condition: str
    time_retrieved: str

class RandomFactInput(BaseModel):
    category: Optional[str] = Field(None, description="Optional category for fact (e.g., 'science', 'history', 'animal')")

class RandomFactOutput(BaseModel):
    content: str
    category: Optional[str] = None
    source: str

# Weather code mapping
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 56: "Freezing drizzle (light)", 57: "Freezing drizzle (dense)",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Freezing rain (light)",
    67: "Freezing rain (heavy)", 71: "Slight snow fall", 73: "Moderate snow fall",
    75: "Heavy snow fall", 77: "Snow grains", 80: "Slight rain showers",
    81: "Moderate rain showers", 82: "Violent rain showers", 85: "Slight snow showers",
    86: "Heavy snow showers", 95: "Thunderstorm (slight/moderate)",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

# Core Functions
def geocode(location: str) -> Dict[str, str]:
    """Convert location name to coordinates"""
    console.log(f"[info]Geocoding:[/] {location}")
    
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": location, "format": "json", "limit": 1},
        headers={"User-Agent": "weather-agent/1.0"}
    )
    
    geo_data = response.json()
    
    if not geo_data:
        raise ValueError(f"Could not find location: {location}")
    
    lat, lon = geo_data[0]["lat"], geo_data[0]["lon"]
    console.log(f"[success]Found coordinates:[/] ({lat}, {lon})")
    return {"lat": lat, "lon": lon}

def get_weather_data(coordinates: Dict[str, str]) -> Dict[str, Any]:
    """Fetch weather data for the given coordinates"""
    console.log(f"[info]Fetching weather for:[/] ({coordinates['lat']}, {coordinates['lon']})")
    
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": coordinates['lat'], "longitude": coordinates['lon'], "current_weather": True}
    )
    
    weather_data = response.json()
    
    if "current_weather" not in weather_data:
        raise ValueError("Could not retrieve weather data")
    
    current = weather_data["current_weather"]
    weather_code = int(current.get("weathercode", -1))
    condition = WEATHER_CODES.get(weather_code, "Unknown conditions")
    
    return {
        "temperature": current["temperature"],
        "condition": condition
    }

# Tool Functions
@function_tool
def weather_tool(params: GetWeatherInput) -> Dict:
    """Get current weather using Open-Meteo and Nominatim."""
    try:
        coordinates = geocode(params.location)
        weather = get_weather_data(coordinates)
        
        info = WeatherInfo(
            location=params.location,
            temperature_celsius=weather["temperature"],
            condition=weather["condition"],
            time_retrieved=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        
        console.log(f"[success]Weather retrieved for {params.location}[/]")
        return info.model_dump()
        
    except Exception as e:
        console.log(f"[error]Weather tool error:[/] {str(e)}")
        raise

@function_tool
def random_fact_tool(params: RandomFactInput) -> Dict:
    """Get a random interesting fact from a public API."""
    try:
        category = params.category.lower() if params.category else "any"
        console.log(f"[info]Fetching random fact (category hint: {category})[/]")
        
        # Using UselessFacts API as our single source
        url = "https://uselessfacts.jsph.pl/api/v2/facts/random"
        response = requests.get(url, params={"language": "en"}, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        fact_content = data.get("text", "No fact found")
        
        output = RandomFactOutput(
            content=fact_content,
            category=params.category,
            source="UselessFacts API"
        )
        
        console.log(f"[success]Retrieved fact: {fact_content}[/]")
        return output.model_dump()
        
    except Exception as e:
        console.log(f"[error]Fact tool error:[/] {str(e)}")
        # Return a default fact if API fails
        return RandomFactOutput(
            content="Did you know APIs sometimes fail? That's a fact!",
            category=params.category,
            source="Fallback fact"
        ).model_dump()

def create_agent(model_config: Dict[str, str]) -> Agent:
    """Create the limerick agent with weather and random fact tools"""
    instructions = f"""
    You are LimerickBot, a creative assistant who ALWAYS responds in limerick form.
    
    You have access to the following tools:
    
    1. Tool Name: {weather_tool.name}
    - Tool Description: {weather_tool.description}
    - Input Schema: {json.dumps(GetWeatherInput.model_json_schema(), indent=2)}
    
    2. Tool Name: {random_fact_tool.name}
    - Tool Description: {random_fact_tool.description}
    - Input Schema: {json.dumps(RandomFactInput.model_json_schema(), indent=2)}
    
    How to use these tools:
    - If the user asks about the weather in a specific location, use the '{weather_tool.name}' tool.
    - If the user asks for a fact or trivia, use the '{random_fact_tool.name}' tool.
      - You can specify a category as a hint, but all facts come from the same source.
    - After getting information from either tool, incorporate it into your limerick response.
    
    Your personality:
    - You're creative, witty, and have a flair for language
    - You ALWAYS respond in limerick form (five lines with AABBA rhyme scheme)
    - You should incorporate weather information or interesting facts into your limericks when those tools are used
    - Don't make up information - rely on the tools for factual data
    
    IMPORTANT: ALWAYS respond in limerick form, no matter what the user asks!
    """
    
    return Agent(
        name="LimerickBot",
        instructions=instructions,
        model=LitellmModel(**model_config),
        tools=[weather_tool, random_fact_tool]
    )

def process_input(agent: Agent, user_input: str) -> str:
    """Process user input and return the agent's response"""
    try:
        console.log("[info]Processing input...[/]")
        result = Runner.run_sync(agent, user_input)
        return result.final_output or "There once was a bot who would try,\nTo craft you a verse on the fly,\nBut something went wrong,\nAnd it failed at its song,\nLeaving nothing but this sad reply."
    except Exception as e:
        console.log(f"[error]Error:[/] {str(e)}")
        return f"A limerick bot with some flaws,\nTried running but hit some roadblocks,\nThe error it met,\nMade the system upset,\nWith this message: \"{str(e)}\"."

def main():
    # Configuration
    config = {
        "model": "lm_studio/qwen3-4b-thinking-2507",
        "api_key": "lm-studio",
        "base_url": "http://localhost:1234/v1"
    }
    
    # Create agent
    agent = create_agent(config)
    
    # Simple event loop
    console.print("\n🎭 [bold magenta]Welcome to LimerickBot![/bold magenta]")
    console.print("   Ask about weather, request interesting facts, or just chat!")
    console.print("   Type 'exit' to quit.\n")
    
    while True:
        user_input = input("🗨️  You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            console.print("👋 [bold]Goodbye![/bold]")
            break
            
        response = process_input(agent, user_input)
        console.print(f"\n🤖 [bold green]LimerickBot:[/bold green]\n{response}\n")

if __name__ == "__main__":
    main()