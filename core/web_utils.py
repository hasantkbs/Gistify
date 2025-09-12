import requests
from bs4 import BeautifulSoup

def get_text_from_url(url: str) -> str:
    """
    Fetches the content from a URL and extracts the text.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes

        # Use BeautifulSoup to parse the HTML and extract text
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text
        text = soup.get_text()

        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text

    except requests.exceptions.RequestException as e:
        # Handle network-related errors
        raise Exception(f"Error fetching URL: {e}")
    except Exception as e:
        # Handle other errors (e.g., parsing)
        raise Exception(f"Error processing URL content: {e}")

