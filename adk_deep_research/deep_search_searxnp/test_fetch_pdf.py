from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy
import asyncio

import requests
from PyPDF2 import PdfReader
from io import BytesIO

async def process_url(url):
    # Step 1: Check Content-Type
    # async with aiohttp.ClientSession() as session:
    #     async with session.head(url, allow_redirects=True) as response:
    #         content_type = response.headers.get("Content-Type", "")
    
    # Step 2: Crawl with Crawl4AI if relevant
    # config = BrowserConfig(accept_downloads=True)

    ispdf = True
    # async with AsyncWebCrawler() as crawler:
    #     result = await crawler.arun(url=url)
    #     print(result.response_headers['content-type'])
    #     if result.response_headers['content-type'] == "application/pdf":
    #         print("PDF detected")
    #         ispdf = True
    #     else:
    #     # print(result.markdown)  # Access extracted text
    #         print(result.metadata)  # Access PDF metadata (title, author, etc.)
    #         return result.markdown
    if ispdf:
        # async with AsyncWebCrawler(crawler_strategy=PDFCrawlerStrategy()) as crawler:
        #     result = await crawler.arun(url=url, config=CrawlerRunConfig(scraping_strategy=PDFContentScrapingStrategy()), cache_mode=CacheMode.BYPASS)
        #     # print(result.markdown)  # Access extracted text
        #     print(result.metadata)  # Access PDF metadata (title, author, etc.)
        #     return result.markdown
        # Fetch the PDF
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
            print(full_text)
        else:
            print("Failed to fetch PDF")

url = "https://www.broadinstitute.org/files/shared/mia/MIASeminarApril212016Rivas_MultipleTestingandFDR.pdf" #"https://openreview.net/pdf?id=md68e8iZK1"  # Replace with your URL
# url = "https://docs.crawl4ai.com/blog/releases/0.5.0/"
asyncio.run(process_url(url))