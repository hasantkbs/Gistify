# Gistify Python SDK

This is the official Python SDK for the Gistify API.

## Installation

```bash
pip install gistify
```

## Usage

```python
from gistify import GistifyClient

client = GistifyClient(base_url="http://localhost:8000")

# Login to get an access token
client.login("test@example.com", "password")

# Summarize a piece of text
summary_result = client.summarize_text("Your long text goes here...")
print(summary_result)
```
