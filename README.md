# Gistify: Smart Summarization Application

Gistify is a smart summarization application that takes inputs in various formats (text, PDF, DOCX, and soon audio) and summarizes them. It leverages advanced summarization models using the `transformers` library and employs advanced chunking and recursive summarization strategies for long texts. The application provides summarization services through a modular API.

## Features

- **Multi-Input Support:** Ability to summarize text, PDF, and DOCX files. (Audio input support will be added in the future).
- **Modular Architecture:** An easily extensible and maintainable structure.
- **FastAPI-powered API:** Provides summarization services through a modern and fast web API.
- **High-Quality Summaries:** Utilizes powerful transformer models like `facebook/mbart-large-50-many-to-many-mmt`.
- **Long Text Processing:** Advanced chunking and recursive summarization strategies for long texts.

## Project Structure

```
Gistify/
├───main.py             # Entry point for starting the FastAPI application
├───config/             # Configuration files
│   └───settings.py
├───core/
│   ├───summarizer.py   # Core summarization logic
│   └───utils.py        # General utility functions (text cleaning, chunking)
├───input_handlers/
│   ├───text_handler.py # Reading text files
│   ├───pdf_handler.py  # PDF text extraction
│   ├───docx_handler.py # DOCX text extraction
│   └───audio_handler.py# Audio input processing (future)
├───api/
│   ├───main.py         # FastAPI application instance
│   └───routes.py       # API endpoints
├───tests/              # Unit and integration tests
├───requirements.txt    # Project dependencies
└───README.md           # This file
```

## Setup

Follow these steps to run this project in your local environment:

1.  **Ensure Python 3.8+ and pip are installed.**

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # or venv\\Scripts\\activate # Windows
    ```

3.  **Install Project Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

The application runs as a FastAPI service. To start the service:

```bash
uvicorn main:app --reload
```

The service will typically run at `http://127.0.0.1:8000`.

### API Endpoints

#### Summarize Text

`POST /summarize`

**Input:**

```json
{
  "text": "Long text to be summarized goes here."
}
```

**Example `curl` command:**

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
     -H "Content-Type: application/json" \
     -d '{"text": "This is an example of a very long text to be summarized. Gistify will take this text and generate a concise summary capturing its main ideas. This helps users quickly understand large amounts of information."}'
```

#### Summarize File

`POST /summarize_file`

**Input:**

```json
{
  "file_path": "/path/to/your/document.pdf"
}
```

**Supported File Formats:** `.txt`, `.pdf`, `.docx`

**Example `curl` command:**

```bash
# For PDF files
curl -X POST "http://127.0.0.1:8000/summarize_file" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "/Users/hasantekbas/Downloads/Algorix Project Doc/Gistify/Code/Gistify/fetih.pdf"}'

# For TXT files
# curl -X POST "http://127.0.0.1:8000/summarize_file" \
#      -H "Content-Type: application/json" \
#      -d '{"file_path": "/path/to/your/text_document.txt"}'

# For DOCX files
# curl -X POST "http://127.0.0.1:8000/summarize_file" \
#      -H "Content-Type: application/json" \
#      -d '{"file_path": "/path/to/your/word_document.docx"}'
```

## Used Model

This project uses Hugging Face's `facebook/mbart-large-50-many-to-many-mmt` model for summarization. This model is specifically trained for multilingual text summarization and produces high-quality, fluent summaries.

The model will be downloaded automatically the first time it is used. This process may take some time depending on your internet connection and the size of the model.

**Note:** A post-processing step is applied to the summary to reduce English word hallucinations that the model might sometimes produce in Turkish texts.

## Advanced Summarization Strategy

To enable the application to summarize long texts more effectively, a chunking and recursive summarization strategy has been implemented. This strategy:
1.  Divides the input text into smaller chunks.
2.  Summarizes each chunk separately.
3.  Combines these chunk summaries.
4.  If the combined summary is still too long, it summarizes these summaries again to obtain the final summary.

## Error Handling

If an error occurs during summarization (e.g., failure to download the model or processing error), the API will return an appropriate error message.