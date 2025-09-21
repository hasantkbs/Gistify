# Gistify: Smart Summarization API

Gistify is a smart summarization application that takes text or files (PDF, DOCX, TXT) and generates a concise summary. It leverages advanced multilingual summarization models and employs a robust chunking strategy for long documents. The application is served via a modern FastAPI backend.

## Features

- **Question Answering from Summary:** Ask questions about summarized content and get concise, relevant answers.
- **Real-time Summarization:** Summarize live audio/video transcripts or streaming data via WebSockets.
- **Sentiment Analysis Integration:** Get an overall sentiment score or sentiment highlights within the summary.
- **Tone/Style Adjustment:** Generate summaries in a specific tone (e.g., formal, informal, neutral, objective, subjective).
- **Webhooks:** Receive asynchronous notifications about the completion of large document or batch processing jobs.
- **Usage Dashboard (Backend):** API endpoints providing insights into API usage, costs (summaries count), and performance.
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
│   ├───utils.py        # Text cleaning and chunking utilities
│   ├───qa.py           # Question Answering logic
│   ├───sentiment.py    # Sentiment Analysis logic
│   ├───streaming_summarizer.py # Real-time streaming summarization logic
│   └───webhook_dispatcher.py # Webhook dispatching logic
├───input_handlers/
│   ├───text_handler.py # Logic for reading .txt files
│   ├───pdf_handler.py  # Logic for extracting text from PDFs
│   └───docx_handler.py # Logic for extracting text from DOCX files
├───api/
│   ├───main.py         # FastAPI application instance and entry point
│   ├───routes.py       # API endpoints
│   ├───schemas.py      # Pydantic models for API data validation
│   ├───models.py       # SQLAlchemy models for database interaction
│   ├───crud.py         # CRUD operations for database models
│   ├───auth.py         # Authentication and JWT handling
│   └───deps.py         # FastAPI dependency injection utilities
├───tests/
│   └───test_api.py     # API tests
├───gistify_python_sdk/ # Python SDK for Gistify API
│   ├───README.md
│   ├───setup.py
│   └───gistify/
│       └───__init__.py # Python SDK client
├───requirements.txt    # Project dependencies
├───requirements-dev.txt # Dependencies for development and testing
└───README.md           # This file
```

## Setup

Gistify'nin çalışması için gerekli bağımlılıklar `requirements.txt` dosyasında, geliştirme ve test için ek bağımlılıklar ise `requirements-dev.txt` dosyasında listelenmiştir.

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

Summarizes raw text with optional tone adjustment.

**Input Body:**
```json
{
  "text": "Your long text to be summarized goes here.",
  "summary_length": "medium", // Optional: "short", "medium", "long"
  "summary_type": "abstractive", // Optional: "abstractive", "extractive"
  "extractive_method": "textrank", // Optional: "lsa", "lexrank", "textrank" (for extractive only)
  "tone": "neutral" // Optional: "formal", "informal", "neutral", "objective", "subjective"
}
```

**Example `curl`:**
```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
     -H "Content-Type: application/json" \
     -d '{"text": "This is a long text that needs to be summarized. Gistify will process it and return a concise version.", "tone": "formal"}'
```

#### Summarize File
`POST /summarize_file`

Summarizes an uploaded file.

**Input:** `multipart/form-data` with a `file` field.

**Supported File Formats:** `.txt`, `.pdf`, `.docx`

**Example `curl`:**
```bash
curl -X POST "http://127.0.0.1:8000/summarize_file" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/document.pdf" \
     -F "summary_length=medium" \
     -F "summary_type=abstractive" \
     -F "extractive_method=textrank" \
     -F "tone=neutral"
```
*(Replace `/path/to/your/document.pdf` with the actual file path)*

#### Summarize URL
`POST /summarize_url`

Summarizes content from a given URL.

**Input Body:**
```json
{
  "url": "https://example.com/article",
  "summary_length": "medium", // Optional: "short", "medium", "long"
  "summary_type": "abstractive", // Optional: "abstractive", "extractive"
  "extractive_method": "textrank", // Optional: "lsa", "lexrank", "textrank" (for extractive only)
  "tone": "neutral" // Optional: "formal", "informal", "neutral", "objective", "subjective"
}
```

**Example `curl`:**
```bash
curl -X POST "http://127.0.0.1:8000/summarize_url" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://www.example.com/news-article", "summary_length": "short"}'
```

#### Summarize Multiple Documents
`POST /summarize_multiple`

Summarizes multiple text documents in a batch.

**Input Body:**
```json
{
  "texts": [
    "First document text...",
    "Second document text..."
  ],
  "summary_length": "medium", // Optional: "short", "medium", "long"
  "summary_type": "abstractive", // Optional: "abstractive", "extractive"
  "extractive_method": "textrank", // Optional: "lsa", "lexrank", "textrank" (for extractive only)
  "tone": "neutral" // Optional: "formal", "informal", "neutral", "objective", "subjective"
}
```

**Example `curl`:**
```bash
curl -X POST "http://127.0.0.1:8000/summarize_multiple" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Text 1.", "Text 2."]}'
```

#### Question Answering
`POST /qa`

Answers a question based on a provided summary.

**Input Body:**
```json
{
  "question": "What is the main topic?",
  "summary": "The article discusses the impact of AI on society."
}
```

**Example `curl`:**
```bash
curl -X POST "http://127.0.0.1:8000/qa" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is Gistify?", "summary": "Gistify is a smart summarization application."}'
```

#### Real-time Summarization (WebSocket)
`WS /ws/summarize`

Provides real-time summarization for streaming text data.

**Usage:** Connect to this WebSocket endpoint and send text chunks. The server will respond with updated summaries.

**Example (JavaScript using WebSocket API):**
```javascript
const ws = new WebSocket("ws://127.0.0.1:8000/ws/summarize");

ws.onopen = (event) => {
  console.log("WebSocket opened");
  ws.send("This is the first part of a very long text. ");
  ws.send("And this is the second part, continuing the discussion.");
};

ws.onmessage = (event) => {
  console.log("Received summary:", event.data);
};

ws.onclose = (event) => {
  console.log("WebSocket closed");
};

ws.onerror = (error) => {
  console.error("WebSocket error:", error);
};
```

#### Get Task Status
`GET /tasks/{task_id}`

Retrieves the status and result of an asynchronous summarization task.

**Example `curl`:**
```bash
curl -X GET "http://127.0.0.1:8000/tasks/your-task-id"
```

#### Get Usage Statistics
`GET /usage`

Provides insights into the user's API usage.

**Example `curl`:**
```bash
curl -X GET "http://127.0.0.1:8000/usage" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

#### Webhooks
`POST /webhooks`

Registers a new webhook to receive asynchronous notifications.

**Input Body:**
```json
{
  "url": "https://your-webhook-receiver.com/notify",
  "event_type": "summary_completed", // Supported: "summary_completed", "finetune_completed"
  "is_active": true
}
```

**Example `curl`:**
```bash
curl -X POST "http://127.0.0.1:8000/webhooks" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
     -d '{"url": "https://example.com/webhook-receiver", "event_type": "summary_completed"}'
```

`GET /webhooks`

Lists all registered webhooks for the current user.

`DELETE /webhooks/{webhook_id}`

Deletes a specific webhook.
