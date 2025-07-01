"""
Modular web scraping implementation using Crawl4AI.
Supports multiple extraction strategies including LLM, CSS, and XPath.
"""

import asyncio
import os
from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from .extraction_result import ExtractionResult, print_extraction_result
from .basic_web_scraper import ExtractionConfig
from .strategy_factory import StrategyFactory

import wikipediaapi

def clean_markdown_links(text: str, min_quality_score: float = 0.2) -> Tuple[str, float]:
    """
    Clean markdown links and filter low-quality content.
    Returns tuple of (cleaned_text, quality_score)
    """
    # Split by double newlines to preserve paragraph structure
    paragraphs = text.split('\n\n')
    
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        # Preserve code blocks by checking if paragraph contains ``` tags
        if '```' in paragraph:
            cleaned_paragraphs.append(paragraph)
            continue
            
        lines = paragraph.split('\n')
        filtered_lines = []
        for line in lines:
            line = line.strip()
            # Keep headers regardless of length
            if re.match(r'^#{1,6}\s+', line):
                filtered_lines.append(line)
                continue
            
            # Skip common UI/navigation elements
            if re.match(r'^(Share|Trade|More|Buy|Sell|Download|Menu|Home|Back|Next|Previous|\d+\s*(BTC|USD|EUR|GBP)|\w{3}-\w{1,3}|Currency:.*|You (Buy|Spend|Receive)|≈|\d+\.\d+)', line, re.IGNORECASE):
                continue
                
            # Count words before removing markdown
            word_count = len(re.sub(r'\[.*?\]\(.*?\)|!\[.*?\]\(.*?\)|<.*?>', '', line).split())
            
            # Increase minimum word threshold to 12
            if word_count < 12:
                # Check if line only contains markdown patterns or appears to be a currency/trading related line
                cleaned_line = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)|\[.*?\]\(.*?\)|!\[.*?\]\(.*?\)|<.*?>|\d+(\.\d+)?%?|\$\d+(\.\d+)?', '', line).strip()
                if not cleaned_line or len(cleaned_line.split()) < 8:  # If nothing substantial remains, skip this line
                    continue
            
            filtered_lines.append(line)
        
        # Only add paragraph if it has any lines left
        if filtered_lines:
            cleaned_paragraphs.append('\n'.join(filtered_lines))
    
    # Rejoin with double newlines
    cleaned_text = '\n\n'.join(cleaned_paragraphs)
    
    # Get quality score
    # quality_score = predict_educational_value([cleaned_text])[0]
    quality_score = 0.3
    return cleaned_text, quality_score

def filter_quality_content(text: str, min_quality_score: float = 0.2) -> str:
    """
    Filter content based on quality and returns concatenated quality content
    """
    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    
    # Process each paragraph
    quality_content = []
    for paragraph in paragraphs:
        if not paragraph.strip():  # Skip empty paragraphs
            continue
            
        cleaned_text, quality_score = clean_markdown_links(paragraph, min_quality_score)
        if cleaned_text and quality_score >= min_quality_score:
            quality_content.append((cleaned_text, quality_score))
    
    # Debug print
    print(f"Found {len(quality_content)} quality paragraphs out of {len(paragraphs)} total")
    
    if quality_content:
        return "\n\n".join(text for text, _ in quality_content)
    return text  # Return original text if no quality content found

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

class WebScraper:
    """Unified scraper that encapsulates all extraction strategies and configuration"""
    def __init__(
        self, 
        browser_config: Optional[BrowserConfig] = None,
        strategies: List[str] = ['no_extraction'],
        llm_instruction: str = "Extract relevant content from the provided text, only return the text, no markdown formatting, remove all footnotes, citations, and other metadata and only keep the main content",
        user_query: Optional[str] = None,
        debug: bool = False,
        filter_content: bool = False
    ):
        self.browser_config = browser_config or BrowserConfig(headless=True, verbose=True)
        self.debug = debug
        self.factory = StrategyFactory()
        self.strategies = strategies or ['markdown_llm', 'html_llm', 'fit_markdown_llm', 'css', 'xpath', 'no_extraction', 'cosine']
        self.llm_instruction = llm_instruction
        self.user_query = user_query
        self.filter_content = filter_content
        
        # Validate strategies
        valid_strategies = {'markdown_llm', 'html_llm', 'fit_markdown_llm', 'css', 'xpath', 'no_extraction', 'cosine'}
        invalid_strategies = set(self.strategies) - valid_strategies
        if invalid_strategies:
            raise ValueError(f"Invalid strategies: {invalid_strategies}")
            
        # Initialize strategy map
        self.strategy_map = {
            'markdown_llm': lambda: self.factory.create_llm_strategy('markdown', self.llm_instruction),
            'html_llm': lambda: self.factory.create_llm_strategy('html', self.llm_instruction),
            'fit_markdown_llm': lambda: self.factory.create_llm_strategy('fit_markdown', self.llm_instruction),
            'css': self.factory.create_css_strategy,
            'xpath': self.factory.create_xpath_strategy,
            'no_extraction': self.factory.create_no_extraction_strategy,
            'cosine': lambda: self.factory.create_cosine_strategy(debug=self.debug)
        }

    def _create_crawler_config(self) -> CrawlerRunConfig:
        """Creates default crawler configuration"""
        content_filter = PruningContentFilter(user_query=self.user_query) if self.user_query else PruningContentFilter()
        return CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=content_filter
            )
        )
    


    async def scrape(self, url: str) -> Dict[str, ExtractionResult]:
        """
        Scrape URL using configured strategies
        
        Args:
            url: Target URL to scrape
        """
        # Handle Wikipedia URLs
        if 'wikipedia.org/wiki/' in url:
            try:
                content = get_wikipedia_content(url)
                # Create same result for all strategies since we're using Wikipedia content
                return {
                    strategy_name: ExtractionResult(
                        name=strategy_name,
                        success=True,
                        content=content
                    ) for strategy_name in self.strategies
                }
            except Exception as e:
                if self.debug:
                    print(f"Debug: Wikipedia extraction failed: {str(e)}")
                # If Wikipedia extraction fails, fall through to normal scraping
        
        # Normal scraping for non-Wikipedia URLs or if Wikipedia extraction failed
        results = {}
        for strategy_name in self.strategies:
            config = ExtractionConfig(
                name=strategy_name,
                strategy=self.strategy_map[strategy_name]()
            )
            result = await self.extract(config, url)
            results[strategy_name] = result
            
        return results
    
    async def scrape_many(self, urls: List[str]) -> Dict[str, Dict[str, ExtractionResult]]:
        """
        Scrape multiple URLs using configured strategies in parallel
        
        Args:
            urls: List of target URLs to scrape
            
        Returns:
            Dictionary mapping URLs to their extraction results
        """
        # Create tasks for all URLs
        tasks = [self.scrape(url) for url in urls]
        # Run all tasks concurrently
        results_list = await asyncio.gather(*tasks)
        
        # Build results dictionary
        results = {}
        for url, result in zip(urls, results_list):
            results[url] = result
            
        return results

    async def extract(self, extraction_config: ExtractionConfig, url: str) -> ExtractionResult:
        """Internal method to perform extraction using specified strategy"""
        try:
            config = self._create_crawler_config()
            config.extraction_strategy = extraction_config.strategy

            if self.debug:
                print(f"\nDebug: Attempting extraction with strategy: {extraction_config.name}")
                print(f"Debug: URL: {url}")
                print(f"Debug: Strategy config: {config.extraction_strategy}")
                if self.user_query:
                    print(f"Debug: User query: {self.user_query}")

            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                if isinstance(url, list):
                    result = await crawler.arun_many(urls=url, config=config)
                else:
                    result = await crawler.arun(url=url, config=config)

            if self.debug:
                print(f"Debug: Raw result attributes: {dir(result)}")
                print(f"Debug: Raw result: {result.__dict__}")

            # Handle different result formats based on strategy
            content = None
            if result.success:
                if extraction_config.name in ['no_extraction', 'cosine']:
                    # For strategies that return a list of dictionaries
                    # if hasattr(result, 'markdown_v2'):
                    if hasattr(result, 'markdown'):
                        # content = result.markdown_v2.raw_markdown
                        content = result.markdown.raw_markdown
                    elif hasattr(result, 'raw_html'):
                        content = result.raw_html
                    elif hasattr(result, 'extracted_content') and result.extracted_content:
                        if isinstance(result.extracted_content, list):
                            content = '\n'.join(item.get('content', '') for item in result.extracted_content)
                        else:
                            content = result.extracted_content
                    
                    if self.filter_content and content:
                        # from src.opendeepsearch.context_scraping.utils import filter_quality_content
                        content = filter_quality_content(content)
                else:
                    content = result.extracted_content
                    if self.filter_content and content:
                        # from src.opendeepsearch.context_scraping.utils import filter_quality_content
                        content = filter_quality_content(content)

            if self.debug:
                print(f"Debug: Processed content: {content[:200] if content else None}")

            extraction_result = ExtractionResult(
                name=extraction_config.name,
                success=result.success,
                content=content,
                error=getattr(result, 'error', None)  # Capture error if available
            )
            
            if result.success:
                # extraction_result.raw_markdown_length = len(result.markdown_v2.raw_markdown)
                # extraction_result.citations_markdown_length = len(result.markdown_v2.markdown_with_citations)
                extraction_result.raw_markdown_length = len(result.markdown.raw_markdown)
                extraction_result.citations_markdown_length = len(result.markdown.markdown_with_citations)
            elif self.debug:
                print(f"Debug: Final extraction result: {extraction_result.__dict__}")

            return extraction_result

        except Exception as e:
            if self.debug:
                import traceback
                print(f"Debug: Exception occurred during extraction:")
                print(traceback.format_exc())
            
            return ExtractionResult(
                name=extraction_config.name,
                success=False,
                error=str(e)
            )

async def main():
    # Example usage with single URL
    single_url = "https://example.com/product-page"
    scraper = WebScraper(debug=True)
    results = await scraper.scrape(single_url)
    
    # Print single URL results
    for result in results.values():
        print_extraction_result(result)

    # Example usage with multiple URLs
    urls = [
        "https://example.com",
        "https://python.org",
        "https://github.com"
    ]
    
    multi_results = await scraper.scrape_many(urls)
    
    # Print multiple URL results
    for url, url_results in multi_results.items():
        print(f"\nResults for {url}:")
        for result in url_results.values():
            print_extraction_result(result)

if __name__ == "__main__":
    asyncio.run(main())
