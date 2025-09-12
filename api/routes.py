from __future__ import annotations
from typing import Union
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, status
from pydantic import BaseModel
import os
import tempfile
from enum import Enum

# RQ imports
from rq import Queue, Connection
from rq.job import Job
from redis import Redis

# Connect to Redis for RQ
redis_conn = Redis()
queue = Queue(connection=redis_conn)

from input_handlers.pdf_handler import read_pdf_text
from input_handlers.docx_handler import read_docx_text
from input_handlers.text_handler import read_text_file
from core.web_utils import get_text_from_url
from core.summarizer import summarize_long_text # Import the summarization function

router = APIRouter()

# --- Enums and Pydantic Models ---

class SummaryLength(str, Enum):
    short = "short"
    medium = "medium"
    long = "long"

class SummaryType(str, Enum):
    abstractive = "abstractive"
    extractive = "extractive"

class ExtractiveMethod(str, Enum):
    lsa = "lsa"
    lexrank = "lexrank"
    textrank = "textrank"

class SummarizeRequest(BaseModel):
    text: str

class UrlRequest(BaseModel):
    url: str

class TaskResponse(BaseModel):
    task_id: str
    status: str

class SummarizeResponse(BaseModel):
    summary: str
    keywords: list[str]
    lang: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Union[SummarizeResponse, None] = None


# --- Task Creation Endpoints ---

@router.post("/summarize", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
def summarize_text(
    request: SummarizeRequest,
    summary_length: SummaryLength = Query(SummaryLength.medium, alias="summary_length"),
    summary_type: SummaryType = Query(SummaryType.abstractive, alias="summary_type"),
    extractive_method: ExtractiveMethod = Query(ExtractiveMethod.lsa, alias="extractive_method"),
):
    """
    Accepts text and starts a background summarization task.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided for summarization.")

    job = queue.enqueue(
        summarize_long_text,
        request.text,
        length=summary_length.value,
        summary_type=summary_type.value,
        extractive_method=extractive_method.value
    )
    return {"task_id": job.id, "status": job.get_status()}

@router.post("/summarize_file", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def summarize_file(
    file: UploadFile = File(...),
    summary_length: SummaryLength = Query(SummaryLength.medium, alias="summary_length"),
    summary_type: SummaryType = Query(SummaryType.abstractive, alias="summary_type"),
    extractive_method: ExtractiveMethod = Query(ExtractiveMethod.lsa, alias="extractive_method"),
):
    """
    Accepts a file and starts a background summarization task.
    """
    file_extension = os.path.splitext(file.filename)[1].lower()
    supported_formats = [".txt", ".pdf", ".docx"]

    if file_extension not in supported_formats:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")

    text = ""
    try:
        if file_extension == ".txt":
            text = read_text_file(tmp_path)
        elif file_extension == ".pdf":
            text = read_pdf_text(tmp_path)
        elif file_extension == ".docx":
            text = read_docx_text(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not text:
        raise HTTPException(status_code=400, detail="Could not read text from the file or file is empty.")

    job = queue.enqueue(
        summarize_long_text,
        text,
        length=summary_length.value,
        summary_type=summary_type.value,
        extractive_method=extractive_method.value
    )
    return {"task_id": job.id, "status": job.get_status()}


@router.post("/summarize_url", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
def summarize_url(
    request: UrlRequest,
    summary_length: SummaryLength = Query(SummaryLength.medium, alias="summary_length"),
    summary_type: SummaryType = Query(SummaryType.abstractive, alias="summary_type"),
    extractive_method: ExtractiveMethod = Query(ExtractiveMethod.lsa, alias="extractive_method"),
):
    """
    Accepts a URL and starts a background summarization task.
    """
    try:
        text = get_text_from_url(request.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from the URL.")

    job = queue.enqueue(
        summarize_long_text,
        text,
        length=summary_length.value,
        summary_type=summary_type.value,
        extractive_method=extractive_method.value
    )
    return {"task_id": job.id, "status": job.get_status()}


# --- Task Status Endpoint ---

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """
    Retrieves the status and result of an RQ task.
    """
    job = Job.fetch(task_id, connection=redis_conn)
    
    response = {
        "task_id": job.id,
        "status": job.get_status(),
        "result": None
    }

    if job.is_finished: # Use is_finished for successful completion
        response["result"] = job.result
    elif job.is_failed: # Use is_failed for failed tasks
        # RQ stores exception info in job.exc_info
        response["result"] = {"summary": job.exc_info, "keywords": [], "lang": ""}
        # To avoid exposing internal errors, you could return a generic error message
        # raise HTTPException(status_code=500, detail="Summarization task failed.")

    return response


# --- Other Endpoints ---

@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok"}
