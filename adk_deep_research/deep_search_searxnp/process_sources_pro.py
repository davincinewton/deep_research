from dataclasses import dataclass
from typing import List, Tuple
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy
import asyncio
from html2text import HTML2Text
from PyPDF2 import PdfReader
from io import BytesIO
import wikipediaapi
from transformers import AutoModelForSequenceClassification
from langchain.text_splitter import RecursiveCharacterTextSplitter

import requests


def create_reranker():
    model = AutoModelForSequenceClassification.from_pretrained(
        'jinaai/jina-reranker-v2-base-multilingual',
        torch_dtype="auto",
        trust_remote_code=True,
        use_flash_attn=False,
    )

    model.to('cuda:0') # or 'cpu' if no GPU is available
    model.eval()
    return model

def get_wikipedia_content(url: str) -> str | None:
    """
    Extract content from a Wikipedia URL.
    
    Args:
        url: Wikipedia URL to scrape
        
    Returns:
        str: Page content if found, None otherwise
    """
    wiki = wikipediaapi.Wikipedia(user_agent="opendeepsearch", language='en')
    
    # Extract the page title from URL (everything after /wiki/)
    try:
        title = url.split('/wiki/')[-1]
        page = wiki.page(title)
        if page.exists():
            return page.text
        return None
    except Exception:
        return None

@dataclass
class Source:
    link: str
    html: str = ""
    # Add other relevant fields here

class SourceProcessor:
    def __init__(
        self, 
        top_results: int = 3,
        reranker_threshold = 0.4,
        # strategies: List[str] = ["no_extraction"],
        # filter_content: bool = True,
        # reranker: str = "None"
    ):
        # self.strategies = strategies
        # self.filter_content = filter_content
        # self.scraper = WebScraper(
        #     strategies=self.strategies, 
        #     filter_content=self.filter_content,
        #     debug=False
        # )
        self.top_results = top_results
        self.reranker_threshold = reranker_threshold
        if self.top_results < 1:
            self.chunker = None
            self.reranker = None
        else:
            self.chunker = RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=96,
                add_start_index=True,
                strip_whitespace=True,
                separators=["\n\n", "\n", ".", " ", ""],

            )
            print("Loading jina reranker...")
            self.reranker = create_reranker()
            
        # chunks = text_splitter.split_documents(documents)
        
        # # Initialize the appropriate reranker
        # if reranker.lower() == "jina":
        #     self.semantic_searcher = JinaReranker()
        #     print("Using Jina Reranker")
        # else:  # default to infinity
        #     self.semantic_searcher = InfinitySemanticSearcher()
        #     print("Using Infinity Reranker")

    async def process_sources(
        self, 
        sources: List[dict], # input sources.data["organic"]
        query: str, 
        pro_mode: bool = False
    ) -> List[dict]:
        print("query: ", query)
        try:
            # sources = [source for source in sources if source and ('ugging' not in source['link']) and ('x.com' not in source['link'])]
            valid_sources = self._get_valid_sources(sources)
            if not valid_sources:
                print("no valid source")
                return sources

            if not pro_mode:
                # Check if there's a Wikipedia article among valid sources
                wiki_sources = [(i, source) for i, source in valid_sources 
                              if 'wikipedia.org' in source['link']]
                if not wiki_sources:
                    print("no wiki source")
                    return sources
                # If Wikipedia article exists, only process that
                valid_sources = wiki_sources[:1]  # Take only the first Wikipedia source
            # print("valid_sources:\n", valid_sources)
            # html_contents = await self._fetch_html_contents([s[1]['link'] for s in valid_sources])
            html_contents = await self._fetch_html_contents([s[1]['link'] for s in valid_sources])
            # print("html_contents:\n", html_contents)
            return self._update_sources_with_content(sources, valid_sources, html_contents, query)
        except Exception as e:
            print(f"Error in process_sources: {e}")
            return sources

    def _get_valid_sources(self, sources: List[dict]) -> List[Tuple[int, dict]]:
        return [(i, source) for i, source in enumerate(sources) if source]
        # return [(i, source) for i, source in enumerate(sources.data['organic'][:num_elements]) if source]

    # async def _fetch_html_contents(self, links: List[str]) -> List[str]:
    #     print("Log:: fetch links->\n", links)
    #     raw_contents = await self.scraper.scrape_many(links)
    #     print(raw_contents)
    #     return [x['no_extraction'].content for x in raw_contents.values()]

    async def _web_to_markdown(self, url) -> str:
        try:
            async with AsyncWebCrawler() as crawler:
                # crawler.warmup()
                # Crawl the URL
                result = await crawler.arun(url=url, wait_for=5, headless=True)

                # Get the HTML content
                html_content = result.html

                # Initialize HTML2Text converter
                h2t = HTML2Text()
                h2t.ignore_links = True  # Keep links in Markdown
                h2t.ignore_images = True  # Keep images in Markdown

                # Convert HTML to Markdown
                # markdown_content = h2t.handle(html_content)
                return h2t.handle(html_content)

        except Exception as e:
            return f"Error processing {url}: {e}"
        
    async def _web_pdf_to_markdown(self, url) -> str:
        try:
            # Handle Wikipedia URLs
            if 'wikipedia.org/wiki/' in url:
                content = get_wikipedia_content(url)
                # Create same result for all strategies since we're using Wikipedia content
                return content
            # If Wikipedia extraction fails, fall through to normal scraping
            ispdf = False
            # deal with arXiv
            if 'arxiv.org/abs' in url:
                url = url.replace("arxiv.org/abs", "arxiv.org/pdf")
                ispdf = True
            if not ispdf:
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url=url)
                    print(result.response_headers['content-type'])
                    if result.response_headers['content-type'] == "application/pdf":
                        print("PDF detected")
                        ispdf = True
                    else:
                        # Get the HTML content
                        html_content = result.html

                        # Initialize HTML2Text converter
                        h2t = HTML2Text()
                        h2t.ignore_links = True  # Keep links in Markdown
                        h2t.ignore_images = True  # Keep images in Markdown

                        # Convert HTML to Markdown
                        # markdown_content = h2t.handle(html_content)
                        return h2t.handle(html_content)
            if ispdf:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Load PDF into memory
                    pdf_file = BytesIO(response.content)
                    
                    # Create a PdfReader object
                    pdf_reader = PdfReader(pdf_file)
                    
                    # Extract text from all pages
                    full_text = ""
                    for page in pdf_reader.pages:
                        full_text += page.extract_text() + "\n\n"
                    
                    # Print or process the extracted text
                    # print(full_text)
                    return full_text
                else:
                    print("Failed to fetch PDF")
                    return ""

        except Exception as e:
            return f"Error processing {url}: {e}"

    # async def _crawl_multiple_urls(self, urls):
    #     tasks = [self._web_to_markdown(url) for url in urls]
    #     results = await asyncio.gather(*tasks, return_exceptions=True)
    #     return results    
    
    async def _crawl_multiple_urls(self, urls, max_concurrent=10):
        semaphore = asyncio.Semaphore(max_concurrent)
        async def sem_web_to_markdown(url):
            async with semaphore:
                return await self._web_pdf_to_markdown(url)
        tasks = [sem_web_to_markdown(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def _fetch_html_contents(self, links: List[str]) -> List[str]:
        print("Fetching links:", links)  # Debug: Inspect links being fetched
        return await self._crawl_multiple_urls(links)


    # async def _fetch_html_contents(self, links: List[str]) -> List[str]:
    #     print("Log:: fetch links->\n", links)
    #     raw_contents = await self.scraper.scrape_many(links)
    #     # raw_contents = await self._crawl_multiple_urls(links)
    #     print(raw_contents)
    #     return [x[self.scraper.strategies[0]].content for x in raw_contents.values()]    

    def _process_html_content(self, html: str, query: str) -> str:
        if not html:
            return ""
        try:
            if self.top_results < 1:
                return html
            # Split the HTML content into chunks
            documents = self.chunker.split_text(html)
            # print("after chunk: ", documents)
            # result = model.rerank(
            #     query,
            #     documents,
            #     max_query_length=512,
            #     max_length=1024,
            #     top_n=3
            # )
            # Rerank the chunks based on the query
            reranked_content = self.reranker.rerank(
                query,
                documents,
                padding='max_length',
                max_query_length=256,
                # max_length=1024,
                overlap = 80,
                top_n=self.top_results
            )
            shrinked_html = ""
            for doc in reranked_content:
                # print(doc)
                if doc['relevance_score'] > self.reranker_threshold:
                    shrinked_html = shrinked_html + doc['document']
            return shrinked_html 
        
        except Exception as e:
            print(f"Error in content processing: {e}")
            return ""

    def _update_sources_with_content(
        self, 
        sources: List[dict],
        valid_sources: List[Tuple[int, dict]], 
        html_contents: List[str],
        query: str
    ) -> List[dict]:
        for (i, source), html in zip(valid_sources, html_contents):
            # print("html before reranking: ", html)
            treranked = self._process_html_content(html, query)
            # print("Reranked html: ",treranked)
            source['html'] = treranked
            sources[i] = source
        return sources