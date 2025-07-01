import logging
import time
import json
from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import LlmAgent, BaseAgent, LoopAgent, SequentialAgent, ParallelAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event, EventActions
# @title 1. Define the before_model_callback Guardrail

# Ensure necessary imports are available
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types # For creating response content
from typing import Optional


from pydantic import BaseModel, Field
import prompts as pt
# from tools import *


API_BASE = 'http://192.168.1.121:8080/v1'
MODEL_ID = "openai/Qwen3-30B-A3B-Q6_K" #"lm_studio/27b"
# API_BASE = 'http://192.168.1.143:1234/v1'
# MODEL_ID  = "lm_studio/27b"

# --- Constants ---
APP_NAME = "deep_research_app"
USER_ID = "rosspanda"
SESSION_ID = "1234"
# GEMINI_2_FLASH = "gemini-2.0-flash"

# os.environ["GOOGLE_API_KEY"] = "AIzaSyCQXjSGS90jbfU8sT8Q5nEEZ2Ec5aj2xgc" # <--- REPLACE
# os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Custom Orchestrator Agent ---
# --8<-- [start:init]
class SearchPlannerAgent(BaseAgent):
    """
    Custom agent for deep research.

    This agent orchestrates a sequence of LLM agents to decompose the requirement of user to a series of queries, then deep search on Webs.
    Check the information retrieved, then summarize.
    Finally, it produce a formal report or output the answer directly if the user prompt is a simple question.
    """

    # --- Field Declarations for Pydantic ---
    # Declare the agents passed during initialization as class attributes with type hints
    # generate tree form of queries
    query_list_generator: LlmAgent
    # query_list_critic
    query_list_critic: LlmAgent
    # revise the tree form of queries
    query_list_reviser: LlmAgent
    # # revise singe query
    # query_reviser: LlmAgent
    # # evaluate the search result meet the requirement of query or not
    # query_result_eval: LlmAgent
    # tree_node_summarizer: LlmAgent
    # final_summarizer: LlmAgent
    loop_agent: LoopAgent
    # sequential_agent: SequentialAgent
    # parralel_agent: ParallelAgent
    # self defined state
    # model_config allows setting Pydantic configurations if needed, e.g., arbitrary_types_allowed
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        query_list_generator: LlmAgent,
        query_list_critic: LlmAgent,
        query_list_reviser: LlmAgent,
        # tone_check: LlmAgent,
    ):
        """
        Initializes the DeepResearAgent.

        Args:
            name: The name of the agent.
            query_list_generator: An LlmAgent to generate list form of queries.
            query_list_critic: An LlmAgent to critique the list form of queries.
            query_list_reviser: An LlmAgent to revise the query list based on criticism.
        """
        # Create internal agents *before* calling super().__init__
        loop_agent = LoopAgent(
            name="CriticReviserLoop", sub_agents=[query_list_critic, query_list_reviser], max_iterations=2
        )
        # sequential_agent = SequentialAgent(
        #     name="PostProcessing", sub_agents=[grammar_check, tone_check]
        # )
        
        # Define the sub_agents list for the framework
        sub_agents_list = [
            query_list_generator,
            loop_agent,
            # sequential_agent,
        ]

        # Pydantic will validate and assign them based on the class annotations.
        super().__init__(
            name=name,
            query_list_generator=query_list_generator,
            query_list_critic=query_list_critic,
            query_list_reviser=query_list_reviser,
            # tone_check=tone_check,
            loop_agent=loop_agent,
            # sequential_agent=sequential_agent,
            sub_agents=sub_agents_list, # Pass the sub_agents list directly
        )
        # self defined state
        
# --8<-- [end:init]

    # --8<-- [start:executionlogic]
    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for the deep research workflow.
        Uses the instance attributes assigned by Pydantic (e.g., self.query_tree_generator).
        """

        logger.info(f"[{self.name}] Starting Deep Research workflow.")

        # 1. Initial Story Generation
        logger.info(f"[{self.name}] Running query_tree_Generator...")
        async for event in self.query_list_generator.run_async(ctx):
            logger.info(f"[{self.name}] Event from QueryTreeGenerator: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        # Check if story was generated before proceeding
        if "query_list" not in ctx.session.state or not ctx.session.state["query_list"]:
             logger.error(f"[{self.name}] Failed to generate query list. Aborting workflow.")
             return # Stop processing if initial story failed

        logger.info(f"[{self.name}] Query list after generator: {ctx.session.state.get('query_list')}")


        # 2. Critic-Reviser Loop
        logger.info(f"[{self.name}] Running CriticReviserLoop...")
        # Use the loop_agent instance attribute assigned during init
        async for event in self.loop_agent.run_async(ctx):
            logger.info(f"[{self.name}] Event from CriticReviserLoop: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        logger.info(f"[{self.name}] Query list after CriticReviserLoop: {ctx.session.state.get('query_list')}")



        # # 3. Sequential Post-Processing (Grammar and Tone Check)
        # logger.info(f"[{self.name}] Running PostProcessing...")
        # # Use the sequential_agent instance attribute assigned during init
        # async for event in self.sequential_agent.run_async(ctx):
        #     logger.info(f"[{self.name}] Event from PostProcessing: {event.model_dump_json(indent=2, exclude_none=True)}")
        #     yield event

        # # 4. Tone-Based Conditional Logic
        # tone_check_result = ctx.session.state.get("tone_check_result")
        # logger.info(f"[{self.name}] Tone check result: {tone_check_result}")

        # if tone_check_result == "negative":
        #     logger.info(f"[{self.name}] Tone is negative. Regenerating story...")
        #     async for event in self.story_generator.run_async(ctx):
        #         logger.info(f"[{self.name}] Event from StoryGenerator (Regen): {event.model_dump_json(indent=2, exclude_none=True)}")
        #         yield event
        # else:
        #     logger.info(f"[{self.name}] Tone is not negative. Keeping current story.")
        #     pass
        #     # logger.info(f"[{self.name}] Tone is not negative. But generate new story again for test.")
        #     # async for event in self.story_generator.run_async(ctx):
        #     #     logger.info(f"[{self.name}] Event from StoryGenerator (Regen): {event.model_dump_json(indent=2, exclude_none=True)}")
        #     #     yield event            

        logger.info(f"[{self.name}] Workflow finished.")
    # --8<-- [end:executionlogic]

# --8<-- [start:llmagents]
# --- Define the individual LLM agents ---
# query tree generator instruction

model_local_text = LiteLlm(model=MODEL_ID, api_base=API_BASE, api_key='lm studio')


query_list_generator = LlmAgent(
    name="QueryListGenerator",
    model=model_local_text,
    # instruction="You are an expert web search planner.",
    input_schema=None,
    # include_contents='none'
    output_key="query_list",  # Key for storing output in session state
)

query_list_critic = LlmAgent(
    name="QueryListCritic",
    model=model_local_text, #GEMINI_2_FLASH,
    instruction="""You are a query list critic. Review the query list provided in
session state with key 'query_list'. Provide 1-2 sentences of constructive criticism
on how to improve it.""",
    input_schema=None,
    output_key="query_list_criticism",  # Key for storing criticism in session state
)

query_list_reviser = LlmAgent(
    name="QueryListReviser",
    model=model_local_text, #GEMINI_2_FLASH,
    
    instruction="""You are a query list reviser. Revise the query list provided in
session state with key 'query_list', based on the criticism in
session state with key 'query_list_criticism'. Output only the revised query list.""",
    # input_schema=None,
    output_key="query_list",  # Overwrites the original story
)

# web_search_searxng = LlmAgent(
#     name="WebSearchSearXng",
#     model=model_local_text,
#     description="Agent to search answer using web search",
#     instruction="You are a helpful web search agent."
#                 "Your task is to search up the answers of a query using 'tool_web_search_searxnp' tool"
#                 "Evaluate Results: Filter results for relevance, credibility, and recency. Cross-reference multiple sources to ensure accuracy and avoid misinformation."
#                 "Summarize Findings: Provide a clear, concise summary of the information, addressing the user's query directly. Include key details, facts, or data points"
#                 ,
#     # input_schema=None,
#     tools=[tool_web_search_searxnp],
#     # include_contents='none',
#     output_key="current_web_result",
# )

# tone_check = LlmAgent(
#     name="ToneCheck",
#     model=GEMINI_2_FLASH,
#     instruction="""You are a tone analyzer. Analyze the tone of the story
# provided in session state with key 'current_story'. Output only one word: 'positive' if
# the tone is generally positive, 'negative' if the tone is generally negative, or 'neutral'
# otherwise.""",
#     input_schema=None,
#     output_key="tone_check_result", # This agent's output determines the conditional flow
# )
# --8<-- [end:llmagents]

# --8<-- [start:story_flow_agent]
# --- Create the custom agent instance ---
search_planner_agent = SearchPlannerAgent(
    name="SearchPlannerAgent",
    query_list_generator=query_list_generator,
    query_list_critic=query_list_critic,
    query_list_reviser=query_list_reviser,
    # grammar_check=grammar_check,
    # tone_check=tone_check,
)

# --- Setup Runner and Session ---
session_service = InMemorySessionService()
initial_state = {"topic": "a brave kitten exploring a haunted house"}
session_running = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
    state=initial_state # Pass initial state here
)
logger.info(f"Initial session state: {session_running.state}")

runner = Runner(
    agent=search_planner_agent, # Pass the custom orchestrator agent
    app_name=APP_NAME,
    session_service=session_service
)

# --- Function to Interact with the Agent ---
def call_agent(user_input_topic: str):
    """
    Sends a new topic to the agent (overwriting the initial one if needed)
    and runs the workflow.
    """
    # current_session = session_service.get_session(app_name=APP_NAME, 
    #                                               user_id=USER_ID, 
    #                                               session_id=SESSION_ID)
    # if not current_session:
    #     logger.error("Session not found!")
    #     return

    # session_running.state["topic"] = user_input_topic
    # --- Define State Changes ---
    current_time = time.time()
    state_changes = {
        "topic": user_input_topic,              # Update session state
    }

    # --- Create Event with Actions ---
    actions_with_update = EventActions(state_delta=state_changes)
    # This event might represent an internal system action, not just an agent response
    system_event = Event(
        invocation_id="inv_topic_update",
        author="system", # Or 'agent', 'tool' etc.
        actions=actions_with_update,
        timestamp=current_time
        # content might be None or represent the action taken
    )

    # --- Append the Event (This updates the state) ---
    session_service.append_event(session_running, system_event)
    print("`append_event` called with explicit state delta.")

    logger.info(f"Updated session state to: {session_running.state}")

    # --- Agent input generate ---
    content = types.Content(role='user', parts=[types.Part(text=f"{pt.Q_T_I}\n# The research topic to plan is:\n{user_input_topic}")])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    final_response = "No final response captured."
    for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            logger.info(f"Potential final response from [{event.author}]: {event.content.parts[0].text}")
            final_response = event.content.parts[0].text

    print("\n--- Agent Interaction Result ---")
    print("Agent Final Response: ", final_response)

    final_session = session_service.get_session(app_name=APP_NAME, 
                                                user_id=USER_ID, 
                                                session_id=SESSION_ID)
    print("Final Session State:")
    import json
    print(json.dumps(final_session.state, indent=2))
    print("-------------------------------\n")
    return final_session.state['query_list']


import asyncio
import os
import re
from deep_search_searxnp import OpenDeepSearchAgent
from json_repair import repair_json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_queries(queries):
    """Process a list of queries asynchronously using deep_search."""
    web_results = []
    sagent = OpenDeepSearchAgent(content_rerank=3)

    for c_q, current_query in enumerate(queries):
        print(f'searching {c_q} query: {current_query}')
        try:
            # Use the async deep_search method
            current_result = await sagent.deep_search(current_query, max_sources=5)
            web_results.append(current_result)
        except Exception as e:
            logger.error(f"Error processing query '{current_query}': {e}")
            web_results.append(f"Error: {e}")
    
    return web_results

if __name__ == "__main__":
    # test topics
    test1 = """Which of the fruits shown in the 2008 painting 'Embroidery from Uzbekistan' were served as part of the October 1949 breakfast menu for the ocean liner that was later used as a floating prop for the film 'The Last Voyage'? Give the items as a comma-separated list, ordering them in clockwise order based on their arrangement in the painting starting from the 12 o'clock position. Use the plural form of each fruit."""
    # --- Run the Agent ---
    # get query_list and set new state current_query, web_result_list, current_web_result as output_key in next search_agent
    tmp_ql = call_agent(test1)
    # print(type(tmp_ql))
    print(tmp_ql)
    # if session_running:
    #     print(session_running.state)
    # else:
    #     print('session not available')
    tmp_ql = repair_json(tmp_ql)
    print(tmp_ql)
    try:
        tmp_ql_j = json.loads(tmp_ql)
        ql = tmp_ql_j.get('queries', 'queries not found')
    except json.JSONDecodeError:
        print("Invalid JSON string")
    # ql = [
    #     "fruits in 2008 painting 'Embroidery from Uzbekistan'",
    #     "October 1949 breakfast menu ocean liner",
    #     "ocean liner used as floating prop for film 'The Last Voyage'",
    #     "clockwise arrangement of fruits in 'Embroidery from Uzbekistan' painting",
    #     "list of fruits served on October 1949 breakfast menu for The Last Voyage prop ship",
    #     "plural form of fruits depicted in 'Embroidery from Uzbekistan'",
    #     "ocean liner used as floating set for 'The Last Voyage' film",
    #     "historical breakfast menus of ocean liners 1940s"
    # ]
    # ql = [
    #     "October 1949 breakfast menu ocean liner",
    # ]

    len_ql = len(ql)
    if len_ql < 1:
        logger.info("Query list empty, stop running!")
        os._exit(0)

    # Run the async query processing
    web_results = asyncio.run(process_queries(ql))

    # Print results (optional)
    for query, result in zip(ql, web_results):
        print(f"\nResults for '{query}':")
        print(result)
    
    # an llm function input query and search result, output <is the query answered?>, <answer> and <is the query vision based?>
    # if the query is not answered and is vision based, then use vision model to output <is the query answered?>, <answer> and <detailed description related with query>

    # collect all the answered query and answer and the original task, input into llm, output the final answer.

    # TBD 02-5-2025: adk structural output, rerank score, web_vision