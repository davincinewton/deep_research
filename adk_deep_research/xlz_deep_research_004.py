import logging
import sys
import json
import ast
import prompts_v2 as pt
# from tools import *

from openai import OpenAI

import asyncio
import os

from json_repair import repair_json
from deep_search_searxnp import OpenDeepSearchAgent, arxiv_search
from web_vision_query_playwright_v3 import web_vision_query
from mdconvert import MarkdownConverter

Normal_Sys = "You are a helpful assistant. Your task is to answer user's question at your best."

# def get_Ini_Message(sprompt:str) ->str:
#     return f"""[
#         {{
#             "role": "system",
#             "content": [{{"type":"text","text": "{sprompt}"}}]
#         }},
#     ]
# """


# API_BASE = 'http://192.168.1.121:8080/v1'
# MODEL_ID = "openai/Qwen3-30B-A3B-Q6_K" #"lm_studio/27b"
API_KEY = "EMPTY"
API_KEY_R = "AIzaSyCQXjSGS90jbfU8sT8Q5nEEZ2Ec5aj2xgc"
API_BASE = 'http://192.168.1.143:8080/v1'
MODEL_ID  = "Qwen3-32B-128K-Q6_K" #"Phi-4-reasoning-plus-UD-Q8_K_XL" #"qwen3-32b" #"QwQ-32B-Q4_K_M" #"Athene-V2-Chat-Q4_K_M" # "Qwen2.5-Coder-32B-Instruct-Q6_K_L" #"Devstral-Small-2505-UD-Q4_K_XL" #"Qwen3-32B-Q4_K_M" #"llama-4-scout-17b-16e-instruct" #
MODEL_ID_G = "gemini-2.0-flash"
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
)

MODEL_ID_1  = "30a" # "phi4p" #"phi-4-reasoning-plus"  #"qwen/qwen3-32b" #"qwen3-14b" #"mistral-small-3.1-24b-instruct-2503" # "27b"# "14b"  # "qwen3-32b" #"gemma-3-27b-it@q8_0" #"qwen3-30b-a3b" #"gemma-3-27b-it-qat" "a72b4b" 
client_1 = OpenAI(
    api_key='lm-studio',
    base_url='http://192.168.1.157:1234/v1',
)


client_r = OpenAI(
    api_key=API_KEY_R,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def inference_with_api(prompt, xmessages:list, model_id, client, sysprompt="You are a helpful assistant."):
    if sysprompt is not None:
        event = {
        "role": "system",
        "content": [{"type":"text","text": sysprompt}],
        }
        xmessages.append(event)
        # print(xmessages)
    event = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
    }
    xmessages.append(event)
    # print(xmessages)
    completion = client.chat.completions.create(
        model = model_id,
        messages = xmessages,
        # seed=7896456457,
        # temperature=0.7,
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
    # tmp = get_Ini_Message(Normal_Sys)
    # print(tmp)
    messages = [] #ast.literal_eval(get_Ini_Message(Normal_Sys))
    # print(messages)
    # print(messages)
    # create
    response, messages = inference_with_api(text_input, messages, model_id=modelid, client=clientapi, sysprompt=pt.Q_T_I)
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

def run_finalizer(text_input: str, task:str, modelid=MODEL_ID,clientapi=client,with_revisor=False):
    messages = [] #ast.literal_eval(get_Ini_Message(Normal_Sys))
    # print(messages)
    # create
    response, messages = inference_with_api(text_input, messages, model_id=modelid, client=clientapi, sysprompt=pt.R_F_I)
    print(response)
    # print(messages)
    if with_revisor:
        # critic
        response, messages = inference_with_api("user's question is: " + task, messages, model_id=modelid, client=clientapi, sysprompt=pt.R_F_C)
        print(response)
        # # print(messages)
        # # revise
        response, messages = inference_with_api("user's question is: " + task, messages, model_id=modelid, client=clientapi, sysprompt=pt.R_F_R)
        print(response)   
    # print(messages)
    tmp = response.find("</think>")
    # print(tmp)
    if tmp != -1:
        response = response[tmp+8:]
        # print(response)
    response = repair_json(response)    
    return response

def run_planner(text_input:str,  modelid=MODEL_ID,clientapi=client):
    # print(tprompt)
    # tmp = get_Ini_Message(Normal_Sys)
    # print(tmp)
    # messages = ast.literal_eval(get_Ini_Message(Normal_Sys))
    # messages = ast.literal_eval(get_Ini_Message("You are an expert of reasoning."))
    messages = []
    # print(messages)
    # print(messages)
    # create
    response, messages = inference_with_api(text_input, messages, model_id=modelid, client=clientapi, sysprompt=pt.P_I)
    # response, messages = inference_with_api(tprompt, messages, model_id=modelid, client=clientapi, sysprompt=Normal_Sys)
    print(response)
    tmp = response.find("</think>")
    # print(tmp)
    if tmp != -1:
        response = response[tmp+8:]
    # tprompt = pt.P_I_C
    # tsys = """ 
    # # print(messages)
    # # revise
    # tprompt = pt.P_I_R
    # tprompt = "You are an expert revisor, based on previous critisms and suggestion, output the revised answer.\n\nuser prompt is:\n"
    # response, messages = inference_with_api(tprompt + text_input, messages, model_id=modelid, client=clientapi)
    # print(response)  

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

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

Black_List = ['ugging', 'linkedin', 'x.com', 'codalab', 'facebook', 'youtube', 'tiktok']
V_Black_List = ['.pdf']

def deep_research(test1:str, with_critic=False)->str:

    ql = [] # query list
    aql = [] # aquery list
    # qql = [] # question list
    # queries generation
    print("first batch query list generating...")
    for iter in range(3):
        # tmp_ql = run_query_generator_with_critic(test1,modelid=MODEL_ID_1,clientapi=client_1)
        # tmp_ql = run_query_generator_with_critic(test1,modelid=MODEL_ID,clientapi=client)
        tmp_ql = run_query_generator_with_critic(test1,modelid=MODEL_ID_G,clientapi=client_r)
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
    # return run_finalizer(all_results_str,test1,modelid=MODEL_ID_1,clientapi=client_1)
    run_finalizer(all_results_str, test1, modelid=MODEL_ID,clientapi=client, with_revisor=with_critic)
    # run_finalizer(all_results_str,test1,modelid=MODEL_ID_G,clientapi=client_r)

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
    question_no = 93 #92 #104 #28 #
    print('load questions: ', question_no)
    file_path = r'G:\work\smolagent\gaia\metadata_test.jsonl'
    jsonl_data = load_jsonl(file_path)
    print(f"{len(jsonl_data)} questions loaded.")
    # print(jsonl_data[29])
    test1 = jsonl_data[question_no].get('Question')
    print("Question: ", test1)
    filename = jsonl_data[question_no].get('file_name')
    filename = filename.strip()
    print("Required file: ", filename)
    if len(filename) < 1:
        attachment = None
    else:
        filename = 'G:\\work\\smolagent\\gaia\\' + filename
        md_converter = MarkdownConverter()
        fileresult = md_converter.convert(filename)
        # print(fileresult.text_content)
        attachment = fileresult.text_content
        test1 +=  "\n\nDocument in Markdown format: \n\n```" + attachment + "```"
    print('\nagent started...\n')

    tooluse = run_planner(test1, modelid=MODEL_ID,clientapi=client)
    # tooluse = run_planner(test1, attachment, modelid=MODEL_ID_1,clientapi=client_1)
    # tooluse = run_planner(test1, attachment, modelid=MODEL_ID_G,clientapi=client_r)
    # tooluse = run_planner(test1,attachment=fileresult.text_content, modelid=MODEL_ID,clientapi=client)
    # tooluse = run_planner(test1,modelid=MODEL_ID_G,clientapi=client_r)
    # os._exit(0)
    # print(tooluse)
    try:
        tooluse = json.loads(tooluse)
        tooluse = tooluse.get('tools', [1])
        if len(tooluse) < 1:
            tooluse = [1]
    except json.JSONDecodeError:
        print("Invalid tooluse JSON string")    
        tooluse = [1]
    # os._exit(0)
    for tool in tooluse:
        if tool == 1:
            print('start reasoning agent...')
            run_finalizer(test1, test1, modelid=MODEL_ID_1,clientapi=client_1)
            break
        if tool == 2:
            print('start deep search agent...')
            deep_research(test1, with_critic=True)
        if tool == 3:
            break

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

    V4 integrate local file load, add planner, add code execute sandbox
    关于 ui：
    网页的形式，分交互区，图像显示区，文字显示区, 在v5实现
    reddit image in base64 coder embeded in url

    23-05-2025 inference function add sys prompt, 如果提供 sys prompt就增加

    1st run phi4p tool call 19 mins
    2st run phi4p simple call 40 mins not finish
    3rd run qwen 3 32b 12 mins simple call
    4th run qwen 3 32b 13 mins tool call
    re run 4th run qwen 3 32b 16 mins tool call
    6th run qwen 3 32b 26 mins simple call model reloaded
    7,8th run llama4 0.5 min, all fail
    9, run phi4p 8bit tool call  mins 1540 1555
    10, run phi4p 8bit simple call  mins 1607 1621
    11, run qwen 3 14b 7mins simple call
    12, run qwen 3 14b 8mins tool call, wrong
    13, run phi4p 8bit tool call 17 mins 
    14th run qwen 3 32b 7 mins tool call
    15 run qwen 3 32b 12 mins simple call
    16, run phi4p 8bit 2gpus simple call 11 mins
    17 run qwen 3 32b 14 mins simple call model new load, wrong
    18, run phi4p 8bit 2gpus simple call 9 mins
    19, run phi4p 8bit tool call 13 mins
    20, run phi4p 6bit 3 gpus tool call  27 mins cancelled
    21, run phi4p 8bit tool call 7.5 mins
    22, run phi4p 8bit 2gpus simple call 12 mins
    23, run phi4p 4bit 3gpus simple call 18 mins
    24, run phi4p 4bit 3gpus tool call 16 mins
    25, run phi4p 4bit 3gpus tool call 20 mins
    26, run phi4p 4bit 3gpus simple call 16 mins
    27 run qwen 3 30b 8 mins simple call
    28 run qwen 3 30b 8 mins tool call, wrong
    29, run phi4p 4bit tool call 21  mins

    结论phi4plus用来做reasoning task. qwen3 32b 作为planner
    
    '''