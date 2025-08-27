# Gistify: Smart Summarization API

Gistify is a smart summarization application that takes text or files (PDF, DOCX, TXT) and generates a concise summary. It leverages advanced multilingual summarization models and employs a robust chunking strategy for long documents. The application is served via a modern FastAPI backend.

## Features

- **File Uploads:** Securely summarize `.txt`, `.pdf`, and `.docx` files by uploading them directly to the API.
- **Direct Text Summarization:** Provide raw text and get a summary.
- **Modular Architecture:** Easily extensible and maintainable structure.
- **High-Quality Summaries:** Utilizes powerful transformer models (`facebook/bart-large-cnn` for English, `csebuetnlp/mT5_multilingual_XLSum` for other languages).
- **Long Text Processing:** Advanced chunking and recursive summarization for handling large documents.
- **Health Check:** A `/health` endpoint to monitor service status.

## Project Structure
```
Gistify/
├───main.py             # CLI entry point (for local testing)
├───config/
│   └───settings.py     # Centralized configuration for models
├───core/
│   ├───summarizer.py   # Core summarization logic
│   └───utils.py        # Text cleaning and chunking utilities
├───input_handlers/
│   ├───text_handler.py # Logic for reading .txt files
│   ├───pdf_handler.py  # Logic for extracting text from PDFs
│   └───docx_handler.py # Logic for extracting text from DOCX files
├───api/
│   ├───main.py         # FastAPI application instance and entry point
│   └───routes.py       # API endpoints
├───tests/
│   └───test_api.py     # API tests
├───requirements.txt    # Project dependencies
├───requirements-dev.txt # Dependencies for development and testing
└───README.md           # This file
```

## Setup

1.  **Ensure Python 3.8+ and pip are installed.**

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The application runs as a FastAPI service. To start the development server:

```bash
uvicorn api.main:app --reload
```

The service will be available at `http://127.0.0.1:8000`.

### API Endpoints

#### Health Check
`GET /health`

Checks if the service is running. Returns `{"status": "ok"}`.

**Example `curl`:**
```bash
curl -X GET "http://127.0.0.1:8000/health"
```

#### Summarize Text
`POST /summarize`

Summarizes raw text.

**Input Body:**
```json
{
  "text": "Your long text to be summarized goes here."
}
```

**Example `curl`:**
```bash
curl -X POST "http://127.0.0.1:8000/summarize" 
     -H "Content-Type: application/json" 
     -d '{"text": "This is a long text that needs to be summarized. Gistify will process it and return a concise version."}'
```

#### Summarize File
`POST /summarize_file`

Summarizes an uploaded file.

**Input:** `multipart/form-data` with a `file` field.

**Supported File Formats:** `.txt`, `.pdf`, `.docx`

**Example `curl`:**
```bash
curl -X POST "http://127.0.0.1:8000/summarize_file" 
     -H "Content-Type: multipart/form-data" 
     -F "file=@/path/to/your/document.pdf"
```
*(Replace `/path/to/your/document.pdf` with the actual file path)*

## Running Tests

To run the tests, first install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

Then, run `pytest` from the project's root directory:

```bash
pytest
```
