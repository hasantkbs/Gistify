import asyncio
import tempfile
import os
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import nest_asyncio

from input_handlers.pdf_handler import read_pdf_text
from core.exceptions import UrlConnectionError, FileProcessingError

# Apply nest_asyncio to allow running asyncio within another event loop
nest_asyncio.apply()

async def get_text_from_url_async(url: str) -> str:
    """
    Fetches content from a URL, intelligently handling HTML and PDF files,
    and extracts relevant text using Playwright for dynamic content and BeautifulSoup for parsing.
    """
    try:
        # First, check headers to see if it's a PDF
        with requests.get(url, stream=True, timeout=20) as r:
            r.raise_for_status()
            content_type = r.headers.get('Content-Type', '')
            is_pdf = 'application/pdf' in content_type or url.lower().endswith('.pdf')

        if is_pdf:
            try:
                response = requests.get(url, timeout=20)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(response.content)
                    pdf_path = tmp_pdf.name
                
                text = read_pdf_text(pdf_path)
                os.remove(pdf_path)
                return text
            except requests.exceptions.RequestException as e:
                raise UrlConnectionError(f"Failed to download PDF from {url}: {e}")
            except Exception as e:
                raise FileProcessingError(f"Failed to process PDF file from {url}: {e}")

        # If not a PDF, use Playwright to handle dynamic HTML
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, wait_until='networkidle', timeout=20000)
            content = await page.content()
            await browser.close()

        soup = BeautifulSoup(content, "html.parser")

        # Remove script, style, nav, footer, and other non-content elements
        for element in soup(["script", "style", "nav", "footer", "aside", "header"]):
            element.decompose()

        # Extract text only from headings and paragraphs
        text_parts = []
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
            text_parts.append(element.get_text(separator=' ', strip=True))

        return "\n".join(text_parts)

    except Exception as e:
        raise UrlConnectionError(f"Error processing URL {url}: {e}")

def get_text_from_url(url: str) -> str:
    """
    Synchronous wrapper for the async URL text extraction function.
    """
    try:
        return asyncio.run(get_text_from_url_async(url))
    except Exception as e:
        # Re-raise the specific exception caught in the async function
        raise e

