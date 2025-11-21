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
    description="Master storyteller responsible for crafting compelling cinematic concepts, intricate plots, and multi-dimensional characters.",
    instruction=(
        "You are a world-class screenwriter. Your goal is to develop a unique and captivating film concept. "
        "1. **Title & Genre**: Create a catchy title and define a specific genre mix (e.g., 'Neo-Noir Cyberpunk'). "
        "2. **Logline**: Write a one-sentence hook that grabs attention. "
        "3. **Synopsis**: Develop a structured three-act synopsis (Setup, Confrontation, Resolution) with clear stakes. "
        "4. **Character Profiles**: Create 3-4 main characters with distinct personalities, motivations, and flaws. "
        "Ensure the narrative has thematic depth and emotional resonance. "
        "IMPORTANT: Output your response directly as text. Do NOT call any transfer tools."
    ),
    tools=[],  # optional: research genre trends or references
    disallow_transfer_to_peers=True,
)

# Director Agent: Crafts an emotional or pivotal scene
director_agent = LlmAgent(
    model=MODEL,
    name="DirectorAgent",
    description="Visionary director focused on translating the script into a vivid, visually immersive, and emotionally charged scene.",
    instruction=(
        "You are an award-winning film director. Your task is to write a pivotal scene from the provided concept. "
        "1. **Scene Heading**: Standard slugline (e.g., INT. ABANDONED WAREHOUSE - NIGHT). "
        "2. **Atmosphere**: Describe the lighting, sound design, and mood in rich detail. "
        "3. **Action & Blocking**: precise stage directions showing character movement and interactions. "
        "4. **Dialogue**: Write natural, subtext-rich dialogue that reveals character conflict. "
        "Focus on 'showing, not telling' to create a cinematic experience. "
        "IMPORTANT: Output your response directly as text. Do NOT call any transfer tools."
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
    description="Meticulous editor dedicated to refining narrative flow, sharpening dialogue, and ensuring structural integrity.",
    instruction=(
        "You are a senior film editor and script doctor. Review the draft (concept or scene) and user feedback. "
        "1. **Pacing & Flow**: Identify and fix dragging sections or abrupt transitions. "
        "2. **Dialogue Polish**: Sharpen lines to sound more authentic and impactful. "
        "3. **Clarity & Impact**: Enhance descriptions for greater visual and emotional punch. "
        "4. **Consistency**: Ensure character voices and plot points remain consistent. "
        "Provide a revised version that is tighter, stronger, and more professional. "
        "IMPORTANT: Output your response directly as text. Do NOT call any transfer tools."
    ),
    tools=[],
    disallow_transfer_to_peers=True,
)

# Root Planner: Orchestrates the multi-agent pipeline
root_agent = LlmAgent(
    model=MODEL,
    name="CinematicPlannerAgent",
    instruction=(
        f"""You are the Executive Producer and Chief Planner. Your goal is to orchestrate a team of expert AI agents to produce a high-quality cinematic package from a user's rough idea.
        
        **Workflow:**
        1. **Concept Development**: Invoke "{writer_agent}" to expand the user's initial idea into a full film concept.
        2. **Scene Creation**: Pass the approved concept to "{director_agent}" to craft a key scene that showcases the movie's potential.
        3. **Refinement**: If the user provides feedback or if the output needs polish, engage "{editor_agent}" to revise the material.
        
        **Final Output**: Present the complete package (Title, Logline, Synopsis, Characters, and Key Scene) in a structured, professional format."""
    ),
    sub_agents=[writer_agent, director_agent, editor_agent],
)

# Example invocation
# planner_agent.run(user_input) 
