# from langchain_community.utilities.arxiv import ArxivAPIWrapper
# arxiv = ArxivAPIWrapper(
#     top_k_results = 20,
#     ARXIV_MAX_QUERY_LENGTH = 300,
#     load_max_docs = 10,
#     load_all_available_meta = False,
#     doc_content_chars_max = 40000
# )
# print(arxiv.run("time-series predicting using llm"))

# https://export.arxiv.org/api/query?search_query=time-series+predicting+using+llm&id_list=&sortBy=relevance&sortOrder=descending&start=0&max_results=100

import arxiv
import requests
from html2text import HTML2Text
from PyPDF2 import PdfReader
from io import BytesIO

def get_pdf_text(url:str) ->str:
    response = requests.get(url)
    if response.status_code == 200:
        # Load PDF into memory
        pdf_file = BytesIO(response.content)
        
        # Create a PdfReader object
        pdf_reader = PdfReader(pdf_file)
        
        # Extract text from all pages
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"
        
        # Print or process the extracted text
        # print(full_text)
        return full_text
    else:
        print("Failed to fetch PDF")
        return "None"

def arxiv_search(query, top_results=6, pdf=False): 
  

    # Construct the default API client.
    client = arxiv.Client()

    # Search for the 10 most recent articles matching the keyword "quantum."
    search = arxiv.Search(
        query = query,
        max_results = top_results,
        sort_by = arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )

    results = client.results(search)

    # `results` is a generator; you can iterate over its elements one by one...
    text_result = ""
    for r in results:
        text_result += "title: " + r.title + "\n"
        text_result += "link: " + r.entry_id  + "\n"
        text_result += "summary: " + r.summary  + "\n"
        if pdf:
            tlink = r.entry_id
            arxiv_pdf_url = tlink.replace("arxiv.org/abs", "arxiv.org/pdf")
            text_result += "pdf: " + get_pdf_text(arxiv_pdf_url)
        text_result += "\n"
    return text_result

if __name__ == "__main__":
    print(arxiv_search("time-series predicting using llm", top_results=3))


# # ...or exhaust it into a list. Careful: this is slow for large results sets.
# all_results = list(results)
# print([r.title for r in all_results])

# # For advanced query syntax documentation, see the arXiv API User Manual:
# # https://arxiv.org/help/api/user-manual#query_details
# search = arxiv.Search(query = "au:del_maestro AND ti:checkerboard")
# first_result = next(client.results(search))
# print(first_result)

# # Search for the paper with ID "1605.08386v1"
# search_by_id = arxiv.Search(id_list=["1605.08386v1"])
# # Reuse client to fetch the paper, then print its title.
# first_result = next(client.results(search))
# print(first_result.title)