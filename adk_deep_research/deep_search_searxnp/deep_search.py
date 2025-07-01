from typing import Optional, Dict, List, Literal
from .serp_search import create_search_api
from .process_sources_pro import SourceProcessor
import asyncio
import nest_asyncio

def extract_information(organic_results: List[Dict]) -> List[str]:
    """Extract snippets from organic search results in a formatted string."""
    formatted_results = []
    for item in organic_results:
        if 'snippet' in item:
            result_parts = [
                f"title: {item.get('title', 'N/A')}",
                f"date: {item.get('date', 'N/A')}",
                f"link: {item.get('link', 'N/A')}",
                f"snippet: {item['snippet']}"
            ]
            
            if ('html' in item) and len(item['html'])>10:
                result_parts.append(f"additional information: {item['html']}")
            
            formatted_results.append('\n'.join(result_parts))
    
    return formatted_results
class OpenDeepSearchAgent:
    def __init__(
        self,
        search_provider: Literal["serper", "searxng"] = "searxng",
        serper_api_key: Optional[str] = None,
        searxng_instance_url: Optional[str] = "http://192.168.1.121:8888/search",
        searxng_api_key: Optional[str] = None,
        content_rerank: int = 3,
        rerank_threshold = 0.5,
    ):
        self.serp_search = create_search_api(
            search_provider=search_provider,
            serper_api_key=serper_api_key,
            searxng_instance_url=searxng_instance_url,
            searxng_api_key=searxng_api_key
        )
        self.source_processor = SourceProcessor(top_results=content_rerank, reranker_threshold=rerank_threshold)

    def get_links(self, query: str, max_sources: int = 3):
        sources = self.serp_search.get_sources(query, num_results=max_sources)
        print("get_source_result:", sources.success)
        if sources.success:
            web_results = sources.data["organic"]
            # img_results = sources.data["images"]
            return web_results
        else:
            print('searching fail, quit.')
            return []
        # web_links = [s['link'] for s in sources.data["organic"]]
        # img_links = [s['link'] for s in sources.data["organic"]]

    async def deep_search(self, sresults, query, full_deep=True):

        processed_sources = await self.source_processor.process_sources(
            sresults,
            query,
            pro_mode=full_deep
        )
        result = extract_information(processed_sources)
        return result
 
    def deep_search_sync(
        self,
        query: str,
        max_sources: int = 5,
        full_deep: bool = True,
    ) -> str:
        try:
            # Try getting the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, use nest_asyncio to allow nested execution
                nest_asyncio.apply()
                return loop.run_until_complete(self.deep_search(query, max_sources, full_deep))
        except RuntimeError:
            # If no loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.deep_search(query, max_sources, full_deep))
            finally:
                loop.close()
    


# Run the async main function
if __name__ == "__main__":
    sagent = OpenDeepSearchAgent(content_rerank=3)
    result = sagent.deep_search_sync(query = "Who built Angkor Wat?", max_sources=5, full_deep=True)
    print(result)


# query = "Fastest land animal?"
# query = "How long would a cheetah at full speed take to run the length of Pont Alexandre III?"
# answer = sagent.ask_sync(query, max_sources=2, pro_mode=True)

# 'organic': organic_results,
# 'images': image_results,
# 'topStories': [],  # SearXNG might not have direct equivalent
# 'graph': None,     # SearXNG doesn't provide knowledge graph
# 'answerBox': None, # SearXNG doesn't provide answer box
# 'peopleAlsoAsk': None,
# 'relatedSearches': data.get('suggestions', [])
# print(sources.error)
# print(sources.data["organic"])
# print(sources.data["images"])
# print(sources.data["topStories"])
# print(sources.data["graph"])
# print(sources.data["answerBox"])
# print(sources.data["peopleAlsoAsk"])
# print(sources.data["relatedSearches"])