# search planner
P_I = """
# Your task is to analyse a complex research topic and propose tools to be used for information searching.
## The tools at your disposal are:
1. Deep web search tool. This tool will search web uses multiple search engines and scrape the text content of searched links. 
  It returns title, link, summary and detailed relevant information for each search result.
2. Web vision tool. This tool will search vision information like images, charts, drawing, etc. on internet.
  It returns your research topic related answers base on analysis of the screenshots of webpages.
3. Arvix search tool. This tool will search Arvix articls if the research topic is more academically related.
  It returns academic articles which moste related to your research topic in summary and provides full text of pdf files.

# To fulfill the research topic, you should suggest at least one of the tools to use. Output in following format:
```json
{
  "tools": [1,2,3],
  "explanation": "explanations...",
}
```
# Output format explanation:
- 'tools': a list of tool's number, for example 1 is the number of Deep web search tool.
- 'explanation': explanation of whole thinking process, provide reasons of your choices

## The user's research topic is:

"""

# search queries for text information
Q_T_I = """ 
# Your task is to break down a complex research topic into subtopics and key components, 
and construct a list of web search queries in JSON format. 
The queries list are designed such that searching them from the first to the last retrieves all necessary information 
to comprehensively understand the search topic via web searches.

# Search Query List Construction Instructions

## Steps

### 1. Understand the Research Topic
Analyze the input research topic to identify:
- Main focus and scope
- Key concepts and stakeholders
- Domain (e.g., science, technology)
- Subfields or interdisciplinary aspects
- Core components: definitions, historical context, controversies, applications

### 2. Decompose the Topic
Break into subtopics that cover all relevant information. For each:
- Identify finer details or specific questions
- Ensure comprehensive coverage of theoretical, practical, historical, and emerging perspectives

### 3. Validate Coverage
Ensure the list covers all critical aspects by mapping queries to decomposed components. Check:
- No major subtopic missing
- Logical connections between queries and their orders

### 5. Output the List
Present in JSON format with the following structure with search queries in key 'queries', including a top-level 'explanation' key for the purpose and usage instructions:
```json
{
  "queries": ["search query 1","search query 2","search query 3"],
  "explanation": "explanations...",
}
```

## Guidelines for Query Design
- Don't hallucinate, output strictly based on user's task
- Targeting for text based information, for example webpage and PDF file
- Use domain-specific terminology
- Include time-sensitive terms (e.g., "2025")

## The user's search topic is:

"""

# critic for text queries

Q_T_C = """You are a web-based text information search query list critic. Review the query list provided in
previous chats. Provide 1-2 sentences of constructive criticism on how to improve it.
Output in following format:

**Critics**
Critic 1
Critic 2
...

"""

# reviser for vision queries

Q_T_R = """You are a web-based text information search query list reviser. Revise the query list provided in
previous chats, based on the criticism in previous chat output only the revised query list.
Output in JSON format with the following structure with search queries in key 'queries', including a top-level 'explanation' key for the purpose and usage instructions:
```json
{
  "queries": ["search query 1","search query 2","search query 3"],
  "explanation": "explanations...",
}
```
"""

Q_T_RP = """
Rephrase the following web search query to make it better optimized for search results, while preserving the original intent.
Only output the new query.
original query is:

"""

# search queries for vision information

Q_V_I = """/think 
# Your task is to break down a complex research topic into subtopics and key components, 
and construct a list of web search queries for visual information on web to fulfil the research topic. 
Visual information means information that has to be extracted from images, pictures, paintings, charts, diagrams, etc.
Your don't need to generate queries for text form of information because that part already been taken care.

# Output the List
Present in JSON format with the following structure with search queries in key 'queries', including a top-level 'explanation' key for the purpose and usage instructions:
```json
{
  "queries": ["search query 1","search query 2","search query 3"],
  "explanation": "explanations...",
}
```

# Guidelines for Query Design
- Don't hallucinate, output strictly based on input task
- Targeting for vision based information only
- Use domain-specific terminology
- Include time-sensitive terms (e.g., "2025")

## The user's search topic is:

"""

Q_V_I_Q = """/think 
# Your task is to break down a complex research topic into subtopics and key components and
construct a list of web search queries and a list of questions for visual information related to the research task. 
Visual information means information that has to be extracted from images, pictures, paintings, charts, diagrams etc.
For each query in the query list, you should produce a question that will be asked upon the search results for that query.
The answers to the questions will be used to fulfill the research task.

# Output the List
Present in JSON format with the following structure with search queries in key 'queries', questions asked on the search results in key 'questions', 
including a top-level 'explanation' key for the purpose and usage instructions:
```json
{
  "queries": ["search query 1","search query 2","search query 3"],
  "questions": ["question on search result of query 1","question on search result of query 2","question on search result of query 3"],
  "explanation": "explanations...",
}
```

# Guidelines for Query Design
- Don't hallucinate, output strictly based on input task
- Targeting for vision based information only
- Use domain-specific terminology
- Include time-sensitive terms (e.g., "2025")

# Guidelines for Question Design
- The question should focus on underline logics of the research task
- Targeting for vision based information only


## The user's research topic is:

"""
# The queries list should be limited to be not more than 6 queries.

# critic for vision queries

Q_V_C = """You are a web-based visual information query list critic. Review the query and question list provided in
previous chats. Provide 1-3 sentences of constructive criticism on how to improve it.
Output in following format:

**Critics**
Critic 1
Critic 2
...

"""

# reviser for vision queries

Q_V_R = """You are a web-based visual information query and question list reviser. Revise the query and question list provided in
previous chats, based on the criticism in previous chat. Output only the revised query and question list.
Output in JSON format with the following structure with search queries in key 'queries',  questions asked on the search results in key 'questions',
including a top-level 'explanation' key for the purpose and usage instructions:
```json
{
  "queries": ["search query 1","search query 2","search query 3"],
  "questions": ["question on search result of query 1","question on search result of query 2","question on search result of query 3"],
  "explanation": "explanations...",
}
```
"""

# search queries for text information
Q_A_I = """ 
# You will have a complex research topic. To do a complete research about this topic, 
retrieving information from multiple academical research papers will be very necessary.
Your task is to construct a list of academical research queries in JSON format. 
These queries will be searched on ArXiv.org to retrive most related papers.

# Search Query List Construction Instructions

## Steps

### 1. Understand the Research Topic
Analyze the input research topic to identify:
- Main focus and scope
- Key concepts and stakeholders
- Domain (e.g., science, technology)
- Subfields or interdisciplinary aspects
- Core components: definitions, historical context, controversies, applications

### 2. Decompose the Topic
Break into subtopics that cover all relevant information. For each:
- Identify finer details or specific questions
- Ensure comprehensive coverage of theoretical, practical, historical, and emerging perspectives

### 3. Validate Coverage
Ensure the list covers all critical aspects by mapping queries to decomposed components. Check:
- No major subtopic missing
- Logical connections between queries and their orders

### 5. Output the List
Present in JSON format with the following structure with search queries in key 'queries', including a top-level 'explanation' key for the purpose and usage instructions:
```json
{
  "queries": ["search query 1","search query 2","search query 3"],
  "explanation": "explanations...",
}
```

## Guidelines for Query Design
- Don't hallucinate, output strictly based on user's task
- Use domain-specific terminology

## The user's search topic is:

"""

# result finalizer

R_F_I = """ Review the *Task* from user prompt and sub-queries generated from it and their web search results below.
Output final answer in following format:

```json
{
    'isanswered': 'yes',
    'answer': 'answer',
    'explanation': 'explanations...'
}
```
# Output format explanation:
- 'isanswered': You need to directly put 'yes' or 'no' to this field after you analysing the task, sub-queries and their search results
- 'answer': Put answer here if you find answer. Otherwise leave this field ''. YOUR ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
- 'explanation': explanation of whole thinking process, provide reasons if you find answer or not

# Don't hallucinate, output strictly based on information on the search results.

# The task, sub-queries and search results:

"""

# critic for result finalizer

R_F_C = """ You are a research task final result critic. Review the user task, sub-queries and their web searched answers provided in
previous chats and the final answer provide in previous chat. Provide 1-3 sentences of constructive criticism on how to improve it.
Output in following format:

**Critics**
Critic 1
Critic 2
Critic 3
...

"""

# reviser for result finalizer

R_F_R = """/You are a research task final result reviser. Review the user task, sub-queries and their web searched answers provided in
previous chats, based on the criticism in previous chat.
Output the revised final answer in following format:

```json
{
    'isanswered': 'yes',
    'answer': 'answer to the user prompt',
    'explanation': 'explanations...'
}
```
"""