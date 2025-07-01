import requests

# from smolagents.agents import ToolCallingAgent
from smolagents import CodeAgent, OpenAIServerModel, tool
from xlz_deep_research_005 import deep_reasoning, deep_research

# Choose which LLM engine to use!
model = OpenAIServerModel(
    model_id="Mistral-Small-3.1-24B-Instruct-2503-UD-Q6_K_XL",
    api_base="http://192.168.1.147:8080/v1",
    api_key="EMPTY",
)

model1 = OpenAIServerModel(
    model_id="Qwen3-32B-128K-Q6_K",
    api_base="http://192.168.1.143:8080/v1",
    api_key="EMPTY",
)
# model = TransformersModel(model_id="meta-llama/Llama-3.2-2B-Instruct")

# For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-20240620'
# model = LiteLLMModel(model_id="gpt-4o")


@tool
def tool_deep_reasoning(task:str, info:str)->str:
    """
    Get the reasoning result of task with additional infomation.
    Args:
        task: A string of current task to reasoning.
        info: Additional information for reasoning.
    Returns:
        str: A string of reasoning result of the task.
    """    
    return deep_reasoning(info, task)

@tool
def tool_deep_web_search(task:str) -> str:
    """
    This is a complex and powerful tool. First it will break down a complex research topic into subtopics and key components, 
    then it will deep web searching all the subtopics from text and images. In the end, it will return the final answer after reasoning.
    Args:
        task: A string of current task.
    Returns:
        str: A string of deep searching and reasoning result of the task.
    """   
    return deep_research(task, with_critic=False, second_batch=False)


# If you want to use the ToolCallingAgent instead, uncomment the following lines as they both will work

# agent = ToolCallingAgent(
#     tools=[
#         convert_currency,
#         get_weather,
#         get_news_headlines,
#         get_joke,
#         get_random_fact,
#         search_wikipedia,
#     ],
#     model=model,
# )


agent = CodeAgent(
    tools=[
        tool_deep_reasoning,
        tool_deep_web_search,
    ],
    model=model,
    # stream_outputs=True,
    max_steps=20,
    verbosity_level=3,
    # planning_interval=4,    
)

# Uncomment the line below to run the agent with a specific query


# agent.run("What is the weather in New York?")
# agent.run("Give me the top news headlines")
# agent.run("Tell me a joke")
# agent.run("Tell me a Random Fact")
# agent.run("who is Elon Musk?")

if __name__ == "__main__":
    in_task = """Which of the fruits shown in the 2008 painting 'Embroidery from Uzbekistan' were served as part of the October 1949 breakfast menu for the ocean liner that was later used as a floating prop for the film 'The Last Voyage'? Give the items as a comma-separated list, ordering them in clockwise order based on their arrangement in the painting starting from the 12 o'clock position. Use the plural form of each fruit."""
    in_doc = None # additional info in markdown
    in_images = None
    if in_doc is not None:
        in_task +=  "\n\nDocument in Markdown format: \n\n```" + in_doc + "```"
    result = agent.run(task=in_task,images=in_images)