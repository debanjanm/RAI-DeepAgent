from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from agents.extensions.models.litellm_model import LitellmModel

# MODEL = "gemini-2.0-flash-001"

config = {
    "model": "lm_studio/qwen3-4b-thinking-2507",
    "api_key": "lm-studio",
    "base_url": "http://localhost:1234/v1"
}

MODEL = LitellmModel(**config)

idea_agent = LlmAgent(
    model=MODEL,
    name="IdeaAgent",
    description="Brainstorms creative and exciting weekend travel ideas based on user preferences or requests.",
    instruction="Your goal is to create travel ideas. Use the tool to brainstorm and respond to the user with 3 exciting weekend trip ideas based on the user's request.",
    tools=[google_search],
    disallow_transfer_to_peers=True,
)

refiner_agent = LlmAgent(
    model=MODEL,
    name="RefineAgent",
    description="Reviews provided travel ideas and selects only those estimated to cost under the provided budget for a weekend trip.",
    instruction="Your task is to review the provided trip ideas, keeping ONLY with the ideas likely under the provided budget for a weekend. If none seem to fit, say so and use your search...",
    disallow_transfer_to_peers=True,
)

root_agent = LlmAgent(
    model=MODEL,
    name="PlannerAgent",
    instruction=f"""You are a Trip Planner, coordinating specialist agents.
    Your goal is to plan a budget-friendly weekend trip ideas. For each user's request, follow the below instructions:
    1. First, use "{idea_agent}" to brainstorm ideas based on the user's request.
    2. Then, use "{refiner_agent}" to take these ideas to filter them for the provided budget.
    3. Present the final, refined list to the user along with the budget.
    """,
    sub_agents=[idea_agent, refiner_agent],
)