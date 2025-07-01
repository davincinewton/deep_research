import logging
import sys
import json
import ast
import prompts as pt
# from tools import *

from openai import OpenAI

import asyncio
import os

from json_repair import repair_json
from deep_search_searxnp import OpenDeepSearchAgent
from web_vision_query_playwright import web_vision_query

Normal_Sys = "You are a helpful assistant. "

def get_Ini_Message(sprompt:str) ->str:
    return f"""[
        {{
            "role": "system",
            "content": [{{"type":"text","text": "{sprompt}"}}]
        }},
    ]
"""


# API_BASE = 'http://192.168.1.121:8080/v1'
# MODEL_ID = "openai/Qwen3-30B-A3B-Q6_K" #"lm_studio/27b"
API_KEY = "EMPTY"
API_KEY_R = "AIzaSyCQXjSGS90jbfU8sT8Q5nEEZ2Ec5aj2xgc"
API_BASE = 'http://192.168.1.147:8080/v1'
MODEL_ID  = "Qwen3-32B-128K-Q6_K"
MODEL_ID_G = "gemini-2.0-flash"
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
)

MODEL_ID_1  = "gemma-3-27b-it-qat" #"a72b4b"
client_1 = OpenAI(
    api_key='lm studio',
    base_url='http://192.168.1.157:1234/v1',
)

client_r = OpenAI(
    api_key=API_KEY_R,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def inference_with_api(prompt, xmessages:list, model_id, client):
    event =     {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
    }
    xmessages.append(event)

    completion = client.chat.completions.create(
        model = model_id,
        messages = xmessages,
        # seed=78967,
        temperature=0,
        timeout=6000.0,
    )
    xmessages.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )
    return completion.choices[0].message.content, xmessages


# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def run_planner(tprompt, tmessages, text_input:str, ):
#     messages = tmessages
#     prompt = tprompt + text_input
#     response, messages = inference_with_api(prompt, messages, model_id=MODEL_ID, client=client)
#     print(response)
#     tmp = response.find("</think>")
#     # print(tmp)
#     if tmp != -1:
#         response = response[tmp+8:]
#         print(response)
#     response = repair_json(response)
#     return response, messages

def run_text_search_planner_with_critic(text_input: str):
    tprompt = pt.Q_T_I + text_input
    # tmp = get_Ini_Message(Normal_Sys)
    # print(tmp)
    messages = ast.literal_eval(get_Ini_Message(Normal_Sys))
    # print(messages)
    # print(messages)
    # create
    response, messages = inference_with_api(tprompt, messages, model_id=MODEL_ID_G, client=client_r)
    print(response)
    # print(messages)
    # critic
    tprompt = pt.Q_T_C
    response, messages = inference_with_api(tprompt, messages, model_id=MODEL_ID_G, client=client_r)
    print(response)
    # print(messages)
    # revise
    tprompt = pt.Q_T_R
    response, messages = inference_with_api(tprompt, messages, model_id=MODEL_ID_G, client=client_r)
    print(response)   
    # print(messages)
    tmp = response.find("</think>")
    # print(tmp)
    if tmp != -1:
        response = response[tmp+8:]
        # print(response)
    response = repair_json(response)    
    return response

def run_vision_search_planner_with_critic(text_input: str):
    tprompt = pt.Q_V_I_Q + text_input
    messages = ast.literal_eval(get_Ini_Message(Normal_Sys))
    # print(messages)
    # create
    response, messages = inference_with_api(tprompt, messages, model_id=MODEL_ID_G, client=client_r)
    print(response)
    # print(messages)
    # # critic
    # tprompt = pt.Q_V_C
    # response, messages = inference_with_api(tprompt, messages, model_id=MODEL_ID, client=client)
    # print(response)
    # # print(messages)
    # # revise
    # tprompt = pt.Q_V_R
    # response, messages = inference_with_api(tprompt, messages, model_id=MODEL_ID, client=client)
    # print(response)   
    # print(messages)
    tmp = response.find("</think>")
    # print(tmp)
    if tmp != -1:
        response = response[tmp+8:]
        # print(response)
    response = repair_json(response)    
    return response

def run_finalizer(text_input: str):
    tprompt = pt.R_F_I + text_input
    messages = ast.literal_eval(get_Ini_Message(Normal_Sys))
    # print(messages)
    # create
    response, messages = inference_with_api(tprompt, messages, model_id=MODEL_ID_G, client=client_r)
    print(response)
    # print(messages)
    # # critic
    tprompt = pt.R_F_C
    response, messages = inference_with_api(tprompt, messages, model_id=MODEL_ID_G, client=client_r)
    print(response)
    # # print(messages)
    # # revise
    tprompt = pt.R_F_R
    response, messages = inference_with_api(tprompt, messages, model_id=MODEL_ID_G, client=client_r)
    print(response)   
    # print(messages)
    tmp = response.find("</think>")
    # print(tmp)
    if tmp != -1:
        response = response[tmp+8:]
        # print(response)
    response = repair_json(response)    
    return response

async def process_queries(sagent,queries,max_sources=5):
    """Process a list of queries asynchronously using deep_search."""
    web_results = []
    

    for c_q, current_query in enumerate(queries):
        print(f'searching {c_q} query: {current_query}')
        try:
            # Use the async deep_search method
            current_result = await sagent.deep_search(current_query, max_sources=max_sources)
            web_results.append(current_result)
        except Exception as e:
            logger.error(f"Error processing query '{current_query}': {e}")
            web_results.append(f"Error: {e}")
    
    return web_results

def get_all_vlinks(sagent:OpenDeepSearchAgent,queries,max_sources=1):
    """Process a list of queries asynchronously using deep_search."""
    web_results = []
    

    for c_q, current_query in enumerate(queries):
        print(f'searching {c_q} query: {current_query}')
        try:
            # Use the async deep_search method
            current_result = sagent.get_links(current_query, max_sources=max_sources)
            web_results.append(current_result)
        except Exception as e:
            logger.error(f"Error processing query '{current_query}': {e}")
            web_results.append(f"Error: {e}")
    
    return web_results

def contains_any(string, substrings):
    return any(substring in string for substring in substrings)

black_list = ['ugging', 'linkedin.com', 'x.com', 'codalab.org']

if __name__ == "__main__":
    #
    # test topics
    # test1 = """Which of the fruits shown in the 2008 painting 'Embroidery from Uzbekistan' were served as part of the October 1949 breakfast menu for the ocean liner that was later used as a floating prop for the film 'The Last Voyage'? Give the items as a comma-separated list, ordering them in clockwise order based on their arrangement in the painting starting from the 12 o'clock position. Use the plural form of each fruit."""
    # test1 = "If we assume all articles published by Nature in 2020 (articles, only, not book reviews/columns, etc) relied on statistical significance to justify their findings and they on average came to a p-value of 0.04, how many papers would be incorrect as to their claims of statistical significance? Round the value up to the next integer."
    # test1 = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."
    test1 = "According to github, when was Regression added to the oldest closed numpy.polynomial issue that has the Regression label in MM/DD/YY?"
#     test1 = """
# Please navigate to https://en.wikipedia.org/wiki/Chicago and give me a sentence containing the word "1992" that mentions a construction accident.
# """
    sagent = OpenDeepSearchAgent(content_rerank=3)
    # --- Run the Agent ---
    # get query_list and set new state current_query, web_result_list, current_web_result as output_key in next search_agent
    print("run text planner...\n")
    tmp_ql = run_text_search_planner_with_critic(test1)
    # tmp_ql, _ = run_planner(pt.Q_T_I, Ini_Messages, test1)
    # print(type(tmp_ql))
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
    web_results = asyncio.run(process_queries(sagent, ql, max_sources=4))

    # Print results (optional)
    all_results = [{"*Task*":test1}]
    for query, result in zip(ql, web_results):
        print(f"\nResults for '{query}':")
        print(result)
        all_results.append({"sub-query":query,"web-result":result})
    
    # an llm function input query and search result, output <is the query answered?>, <answer> and <is the query vision based?>
    # if the query is not answered and is vision based, then use vision model to output <is the query answered?>, <answer> and <detailed description related with query>

    # collect all the answered query and answer and the original task, input into llm, output the final answer.

    # TBD 02-5-2025: adk structural output, rerank score, web_vision
    # start vision search
   
    #***Todo let llm decide whether to do vision search***

    print("\nrun vision planner...\n")
    vision_plan = True
    vql = []
    vqql = []
    while vision_plan:
        tmp_ql = run_vision_search_planner_with_critic(test1)
        # os._exit(0)
        # tmp_ql, _ = run_planner(pt.Q_V_I, Ini_Messages, test1)
        print(tmp_ql)
        try:
            tmp_ql_j = json.loads(tmp_ql)
            vql = tmp_ql_j.get('queries', 'queries not found')
            vqql = tmp_ql_j.get('questions', 'questions not found')
            if len(vql) == len(vqql) and len(vql)>1:
                vision_plan = False
            else:
                print('wrong produced query list')
        except json.JSONDecodeError:
            print("Invalid JSON string")

    # len_vql = len(vql)
    # if len_vql < 1:
    #     logger.info("Query list empty, stop running!")
    #     os._exit(0)
    # get links
    vlinks = get_all_vlinks(sagent, vql, max_sources=3)
    visual_results = []
    for vquery, vlink in zip(vqql, vlinks):
        print(f"\nLinks for '{vquery}':")
        print(vlink)
        tmp_results = []
        for link in vlink:
            if contains_any(link, black_list):
                continue
            # tmp = web_vision_query(vquery + '\nthe above query is one of the sub-queries of:\n' +test1, link)
            tmp = web_vision_query(vquery, link)
            print(tmp)
            tmp_results.append(tmp)
        visual_results.append(tmp_results)
    for query, result in zip(vql, visual_results):
        print(f"\nVResults for '{query}':")
        print(result)
        all_results.append({"sub-query":query,"web-result":result})
    
    all_results_str = json.dumps(all_results)
    print('search results save to file...')
    with open('output.txt', 'w') as file:
        file.write(all_results_str)
    print('\nfinal answer preparing...\n')
    run_finalizer(all_results_str)