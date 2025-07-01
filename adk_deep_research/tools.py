from deep_search_searxnp import deep_search_sync, arxiv_search

def tool_web_search_searxnp(query: str) ->str:
    """
    Search 'query' on Web using multiple searching engines and retrieve result from multiple sources.
    Args:
        query (str): The information or question to search for
    Returns:
        str: A string containing search results which including title, link, date, snippet, additional information(if available) of each source.
    """
    return deep_search_sync(query,max_sources=5,full_deep=False,content_rerank=0)