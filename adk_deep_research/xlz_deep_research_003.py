import logging
import sys
import json
import ast
import prompts_v1 as pt
# from tools import *

from openai import OpenAI

import asyncio
import os

from json_repair import repair_json
from deep_search_searxnp import OpenDeepSearchAgent, arxiv_search
from web_vision_query_playwright_v3 import web_vision_query

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
MODEL_ID  = "Qwen3-32B-Q4_K_M" #"Qwen3-32B-128K-Q6_K" #
MODEL_ID_G = "gemini-2.0-flash"
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
)

MODEL_ID_1  = "qwen3-32b" #"gemma-3-27b-it@q8_0" #"qwen3-30b-a3b" #"qwen3-32b" #"gemma-3-27b-it-qat" "a72b4b" 
client_1 = OpenAI(
    api_key='lm-studio',
    base_url='http://192.168.1.143:1234/v1',
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
        # seed=7896456457,
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

def run_query_generator_with_critic(text_input: str, modelid=MODEL_ID,clientapi=client):
    tprompt = pt.Q_T_I + text_input
    # tmp = get_Ini_Message(Normal_Sys)
    # print(tmp)
    messages = ast.literal_eval(get_Ini_Message(Normal_Sys))
    # print(messages)
    # print(messages)
    # create
    response, messages = inference_with_api(tprompt, messages, model_id=modelid, client=clientapi)
    print(response)
    # print(messages)
    # critic
    # tprompt = pt.Q_T_C
    # response, messages = inference_with_api(tprompt, messages, model_id=modelid, client=clientapi)
    # print(response)
    # # print(messages)
    # # revise
    # tprompt = pt.Q_T_R
    # response, messages = inference_with_api(tprompt, messages, model_id=modelid, client=clientapi)
    # print(response)   
    # print(messages)
    tmp = response.find("</think>")
    # print(tmp)
    if tmp != -1:
        response = response[tmp+8:]
        # print(response)
    response = repair_json(response)    
    return response

def run_finalizer(text_input: str, task: str, modelid=MODEL_ID,clientapi=client):
    tprompt = pt.R_F_I + text_input
    messages = ast.literal_eval(get_Ini_Message(Normal_Sys))
    # print(messages)
    # create
    response, messages = inference_with_api(tprompt, messages, model_id=modelid, client=clientapi)
    print(response)
    # print(messages)
    # # critic
    tprompt = pt.R_F_C
    response, messages = inference_with_api(tprompt + task, messages, model_id=modelid, client=clientapi)
    print(response)
    # # print(messages)
    # # revise
    tprompt = pt.R_F_R
    response, messages = inference_with_api(tprompt + task, messages, model_id=modelid, client=clientapi)
    print(response)   
    # print(messages)
    tmp = response.find("</think>")
    # print(tmp)
    if tmp != -1:
        response = response[tmp+8:]
        # print(response)
    response = repair_json(response)    
    return response

async def process_queries(sagent,sresults, queries):
    """Process a list of queries asynchronously using deep_search."""
    web_results = []
    for c_q, result in enumerate(sresults):
        print(f'searching {c_q} query: {queries[c_q]}')
        try:
            # Use the async deep_search method
            current_result = await sagent.deep_search(result, queries[c_q])
            web_results.append(current_result)
        except Exception as e:
            logger.error(f"Error processing query '{queries[c_q]}': {e}")
            web_results.append(f"Error: {e}")
    
    return web_results


# def get_all_vlinks(sagent:OpenDeepSearchAgent,queries, aqueries, max_sources=1):
#     """Process a list of queries asynchronously using deep_search."""
#     w_results = []
#     w_links = []
#     # for c_q, current_query in enumerate(queries):
#     for c_q in range(len(queries)):
#         current_query = queries[c_q]
#         current_aquery = aqueries[c_q]
#         print(f'searching {c_q} query: {current_query}')
#         try:
#             # Use the async deep_search method
#             result1 = sagent.get_links(current_query, max_sources=max_sources)
#             link1 = [s['link'] for s in result1]
#             # print(link1)
#             result2 = sagent.get_links(current_aquery, max_sources=max_sources)
#             for s in result2:
#                 # print('-> ',s['link'])
#                 if s['link'] in link1:
#                     continue
#                 else:
#                     result1.append(s)
#                     link1.append(s['link'])
#         except Exception as e:
#             logger.error(f"Error processing query '{current_query}': {e}")
#             # w_results.append(f"Error: {e}")
#             continue
#         result2 = []
#         for c_l in range(len(link1)):
#             if contains_any(link1[c_l], Black_List):
#                 continue
#             else:
#                 result2.append(result1[c_l])
#         link2 = [s['link'] for s in result2]
#         w_results.append(result2)
#         w_links.append(link2)
    
#     return w_results, w_links

def get_all_vlinks(sagent:OpenDeepSearchAgent,queries, aqueries, max_sources=1):
    """Process a list of queries asynchronously using deep_search."""
    w_results = []
    w_links = []
    w_query = []
    for c_q, current_query in enumerate(queries):
        print(f'searching {c_q} query: {current_query}')
        try:
            # Use the async deep_search method
            current_result = sagent.get_links(current_query, max_sources=max_sources)
            current_link = [s['link'] for s in current_result]
            tmp_result = []
            tmp_link = []
            for link, result in zip(current_link, current_result):
                if contains_any(link, Black_List):
                    continue
                else:
                    tmp_result.append(result)
                    tmp_link.append(link)
            w_results.append(tmp_result)
            w_links.append(tmp_link)
            w_query.append(current_query)
        except Exception as e:
            logger.error(f"Error processing query '{current_query}': {e}")
            # web_results.append(f"Error: {e}")
            continue
    for c_q, current_query in enumerate(aqueries):
        print(f'searching {c_q} query: {current_query}')
        try:
            # Use the async deep_search method
            current_result = sagent.get_links(current_query, max_sources=max_sources)
            current_link = [s['link'] for s in current_result]
            tmp_result = []
            tmp_link = []
            for link, result in zip(current_link, current_result):
                if contains_any(link, Black_List):
                    continue
                else:
                    tmp_result.append(result)
                    tmp_link.append(link)
            w_results.append(tmp_result)
            w_links.append(tmp_link)
            w_query.append(current_query)
        except Exception as e:
            logger.error(f"Error processing query '{current_query}': {e}")
            # web_results.append(f"Error: {e}")
            continue
    return w_results, w_links, w_query

# def get_all_vlinks(sagent:OpenDeepSearchAgent,queries,max_sources=1):
#     """Process a list of queries asynchronously using deep_search."""
#     w_results = []
#     w_links = []
#     # for c_q, current_query in enumerate(queries):
#     for c_q in range(len(queries)):
#         current_query = queries[c_q]
#         # current_aquery = aqueries[c_q]
#         print(f'searching {c_q} query: {current_query}')
#         try:
#             # Use the async deep_search method
#             result1 = sagent.get_links(current_query, max_sources=max_sources)
#             link1 = [s['link'] for s in result1]
#             # print(link1)
#         except Exception as e:
#             logger.error(f"Error processing query '{current_query}': {e}")
#             # w_results.append(f"Error: {e}")
#             continue
#         result2 = []
#         for c_l in range(len(link1)):
#             if contains_any(link1[c_l], Black_List):
#                 continue
#             else:
#                 result2.append(result1[c_l])
#         link2 = [s['link'] for s in result2]
#         w_results.append(result2)
#         w_links.append(link2)
    
#     return w_results, w_links

def contains_any(string, substrings):
    return any(substring in string for substring in substrings)

Black_List = ['ugging', 'linkedin', 'x.com', 'codalab', 'facebook', 'youtube', 'tiktok']
V_Black_List = ['.pdf']

if __name__ == "__main__":
    #
    # test topics
    # test1 = """Which of the fruits shown in the 2008 painting 'Embroidery from Uzbekistan' were served as part of the October 1949 breakfast menu for the ocean liner that was later used as a floating prop for the film 'The Last Voyage'? Give the items as a comma-separated list, ordering them in clockwise order based on their arrangement in the painting starting from the 12 o'clock position. Use the plural form of each fruit."""
    # test1 = "If we assume all articles published by Nature in 2020 (articles, only, not book reviews/columns, etc) relied on statistical significance to justify their findings and they on average came to a p-value of 0.04, how many papers would be incorrect as to their claims of statistical significance? Round the value up to the next integer."
    # test1 = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."
    # test1 = "According to github, when was Regression added to the oldest closed numpy.polynomial issue that has the Regression label in MM/DD/YY?"
#     test1 = """
# Please navigate to https://en.wikipedia.org/wiki/Chicago and give me a sentence containing the word "1992" that mentions a construction accident.
# """
    # test1 = "In terms of geographical distance between capital cities, which 2 countries are the furthest from each other within the ASEAN bloc according to wikipedia? Answer using a comma separated list, ordering the countries by alphabetical order."
    # test1 = "I need to fact-check a citation. This is the citation from the bibliography:\n\nGreetham, David. \"Uncoupled: OR, How I Lost My Author(s).\" Textual Cultures: Texts, Contexts, Interpretation, vol. 3 no. 1, 2008, p. 45-46. Project MUSE, doi:10.2979/tex.2008.3.1.44.\n\nAnd this is the in-line citation:\n\nOur relationship with the authors of the works we read can often be \u201cobscured not by a \"cloak of print\" but by the veil of scribal confusion and mis-transmission\u201d (Greetham 45-46).\n\nDoes the quoted text match what is actually in the article? If Yes, answer Yes, otherwise, give me the word in my citation that does not match with the correct one (without any article)."
    print('\nagent started...\n')
    test1 = "Which contributor to the version of OpenCV where support was added for the Mask-RCNN model has the same name as a former Chinese head of government when the names are transliterated to the Latin alphabet?"
    # test1 = "What integer-rounded percentage of the total length of the harlequin shrimp recorded in Omar Valencfia-Mendez 2017 paper was the sea star fed to the same type of shrimp in G. Curt Fiedler's 2002 paper?"
    # test1 =  "What is the minimum number of page links a person must click on to go from the english Wikipedia page on The Lord of the Rings (the book) to the english Wikipedia page on A Song of Ice and Fire (the book series)? In your count, include each link you would click on to get to the page. Use the pages as they appeared at the end of the day on July 3, 2023."
    # test1 = "My family reunion is this week, and I was assigned the mashed potatoes to bring. The attendees include my married mother and father, my twin brother and his family, my aunt and her family, my grandma and her brother, her brother's daughter, and his daughter's family. All the adults but me have been married, and no one is divorced or remarried, but my grandpa and my grandma's sister-in-law passed away last year. All living spouses are attending. My brother has two children that are still kids, my aunt has one six-year-old, and my grandma's brother's daughter has three kids under 12. I figure each adult will eat about 1.5 potatoes of mashed potatoes and each kid will eat about 1/2 a potato of mashed potatoes, except my second cousins don't eat carbs. The average potato is about half a pound, and potatoes are sold in 5-pound bags. How many whole bags of potatoes do I need? Just give the number."
    ql = [] # query list
    aql = [] # aquery list
    # qql = [] # question list
    # queries generation
    print("first batch query list generating...")
    for iter in range(3):
        tmp_ql = run_query_generator_with_critic(test1,modelid=MODEL_ID_1,clientapi=client_1)
        # tmp_ql = run_query_generator_with_critic(test1,modelid=MODEL_ID,clientapi=client)
        try:
            tmp_ql_j = json.loads(tmp_ql)
            ql = tmp_ql_j.get('queries', [])
            # aql = tmp_ql_j.get('altqueries', [])
            # qql = tmp_ql_j.get('questions', [])
        except json.JSONDecodeError:
            print("Invalid JSON string")
        # len_ql = len(ql)
        # if len(ql) < 1 or len(ql) != len(aql) or len(ql) != len(qql) or len(qql) != len(aql):
        if len(ql) < 1 :
            logger.info("Query list generate error, re-generate...")
        else:
            break
    # alter-queries generation
    print("second batch query list generating...")
    for iter in range(3):
        # tmp_ql = run_query_generator_with_critic(test1,modelid=MODEL_ID_1,clientapi=client_1)
        # tmp_ql = run_query_generator_with_critic(test1,modelid=MODEL_ID_G,clientapi=client_r)
        tmp_ql = run_query_generator_with_critic(test1,modelid=MODEL_ID,clientapi=client)
        try:
            tmp_ql_j = json.loads(tmp_ql)
            aql = tmp_ql_j.get('queries', [])
            # aql = tmp_ql_j.get('altqueries', [])
            # qql = tmp_ql_j.get('questions', [])
        except json.JSONDecodeError:
            print("Invalid JSON string")
        # len_ql = len(ql)
        # if len(ql) < 1 or len(ql) != len(aql) or len(ql) != len(qql) or len(qql) != len(aql):
        if len(aql) < 1 :
            logger.info("Query list generate error, re-generate...")
        else:
            break        
    if len(ql) < 1:
        logger.info("Query list generate error, stop program.")
        os._exit(0)
    # start search tool
    sagent = OpenDeepSearchAgent(content_rerank=5)
    # get links
    wress, wlinks, tql = get_all_vlinks(sagent, ql, aql, max_sources=5)    
    print('\n')
    print(tql)
    print('\n')
    print(wress)
    print('\n')
    print(wlinks)
    print('\n')
    # initiate search result list
    all_results = [{"*Task*":test1}]
    # get text information, run the async query processing
    print('\nretrieve text information...\n')
    web_results = asyncio.run(process_queries(sagent, wress, tql))

    for query, result in zip(tql, web_results):
        print(f"\nResults for '{query}':")
        print(result)
        all_results.append({"sub-query":query,"web-result":result})
    # get images information 
    print('\nretrieve image information...\n')
    visual_results = []
    for vquery, vlink in zip(tql, wlinks):
        print(f"\nLinks for '{vquery}':")
        print(vlink)
        tmp_results = []
        for link in vlink:
            if contains_any(link, V_Black_List):
                continue
            tmp = web_vision_query(vquery, link, num_scroll=1)
            print(tmp)
            tmp_results.append(tmp)
        visual_results.append(tmp_results)
    for query, result in zip(ql, visual_results):
        print(f"\nVResults for '{query}':")
        print(result)
        all_results.append({"sub-query":query,"vision-result":result})    

    all_results_str = json.dumps(all_results)
    print('search results save to file...')
    with open('output.txt', 'w') as file:
        file.write(all_results_str)
    print('\nfinal answer preparing...\n')
    run_finalizer(all_results_str,test1,modelid=MODEL_ID_1,clientapi=client_1)
    # run_finalizer(all_results_str,test1, modelid=MODEL_ID,clientapi=client)
    # run_finalizer(all_results_str,test1,modelid=MODEL_ID_G,clientapi=client_r)

    '''
    v3做成全聚合搜索，arxiv集成到deep search, 像wiki一样在单链接爬虫时将pdf读出来，
    每个搜索配1-2个问题，将图片链接一起下载。爬虫完之后下载图片并识别
    
    Images are scored based on point based system, to filter based on usefulness. Points are assigned
to each image based on the following aspects.

If either height or width exceeds 150px
If image size is greater than 10Kb
If alt property is set
If image format is in jpg, png or webp
If image is in the first half of the total images extracted from the page
Default threshold is 2.

    图片识别怎加评论和复习，或者问题重构，或者多个模型重构

    输出报告report generation. ppt generation

    add back questions list?
    put all results and image back to gemma?

    V3 done in 20-5-2-25


    '''