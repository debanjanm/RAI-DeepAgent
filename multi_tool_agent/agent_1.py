from google.adk.agents import LlmAgent
# from google.adk.tools import text2im, google_search  # add any other tools as needed

# Agent Definitions for AI-Powered Cinematic Storytelling Hackathon
# MODEL = "gemini-2.0-flash-001"
from google.adk.models.lite_llm import LiteLlm
config = {
    "model": "lm_studio/qwen3-4b-thinking-2507",
    "api_key": "lm-studio",
    "base_url": "http://localhost:1234/v1"
}

MODEL = LiteLlm(**config)

# Writer Agent: Crafts the high-level story blueprint
writer_agent = LlmAgent(
    model=MODEL,
    name="WriterAgent",
    description="Generates a cinematic concept including title, genre, synopsis, and characters.",
    instruction=(
        "Your role is to develop the film's core concept. "
        "Provide a title, specify genre, write a concise synopsis, "
        "and outline the key characters with short descriptions."
    ),
    tools=[],  # optional: research genre trends or references
    disallow_transfer_to_peers=True,
)

# Director Agent: Crafts an emotional or pivotal scene
director_agent = LlmAgent(
    model=MODEL,
    name="DirectorAgent",
    description="Writes a pivotal or emotional scene based on the concept.",
    instruction=(
        "Your task is to write a vivid, cinematic scene that captures "
        "the story's emotional high point. Include stage directions, dialogue, and mood details."
    ),
    tools=[],
    disallow_transfer_to_peers=True,
)

# Visualizer Agent: Generates visuals for concept and key scene
# visualizer_agent = LlmAgent(
#     model=MODEL,
#     name="VisualizerAgent",
#     description="Produces visual assets (storyboard frames, concept art) based on the story and scene.",
#     instruction=(
#         "Create prompts and use the text2im tool to generate concept art or storyboard frames "
#         "that reflect the concept and the pivotal scene."
#     ),
#     tools=[text2im],
#     disallow_transfer_to_peers=True,
# )

# Editor Agent: Revises the concept or scene based on feedback
editor_agent = LlmAgent(
    model=MODEL,
    name="EditorAgent",
    description="Revises and polishes the concept or scene after receiving feedback.",
    instruction=(
        "Review the provided concept or scene draft along with user feedback. "
        "Suggest improvements, tighten pacing, and refine language and visual prompts."
    ),
    tools=[],
    disallow_transfer_to_peers=True,
)

# Root Planner: Orchestrates the multi-agent pipeline
root_agent = LlmAgent(
    model=MODEL,
    name="CinematicPlannerAgent",
    instruction=(
        f"""You are the Cinematic Planner coordinating specialized AI agents. Given Rought Draft From User, your goal is to create a cinematic package.
        Follow these steps:
        "1. Invoke "{writer_agent}" to draft the film concept. "
        "2. Pass the concept to "{director_agent}" to write the key scene. "
        "3. After feedback, engage "{editor_agent}" to refine and polish all outputs. "
        "Finally, collect and present the complete cinematic package to the user."""
    ),
    sub_agents=[writer_agent, director_agent, editor_agent],
)

# Example invocation
# planner_agent.run(user_input) 
