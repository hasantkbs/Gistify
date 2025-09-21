from __future__ import annotations
from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, UploadFile, File, Query, status, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
import os
import tempfile
from enum import Enum
from sqlalchemy.orm import Session
import hashlib
import shutil

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
from core.summarizer import summarize_long_text, summarize_multiple_documents
from core.exceptions import (
    UnsupportedFileTypeError,
    FileProcessingError,
    EmptyContentError,
    UrlConnectionError,
)
from core import cache, finetuner # Import the finetuner module
from core.entity_extractor import extract_entities # Import the entity extractor
from core.qa import answer_question_from_summary
from core.streaming_summarizer import StreamingSummarizer
from core.webhook_dispatcher import dispatch_webhook

from . import crud, schemas, auth, deps, models
from .database import get_db

router = APIRouter()
auth_router = APIRouter()
finetune_router = APIRouter()
webhook_router = APIRouter()
qa_router = APIRouter()

@qa_router.post("/", response_model=schemas.QAResponse)
def answer_question(
    request: schemas.QARequest,
    current_user: schemas.User = Depends(deps.get_current_user),
):
    """
    Answers a question based on a given summary.
    """
    if not request.question or not request.summary:
        raise HTTPException(status_code=400, detail="Question and summary must be provided.")

    answer = answer_question_from_summary(request.question, request.summary)
    return answer


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

class Sentiment(BaseModel):
    label: str
    score: float


class SummarizeResponse(BaseModel):
    summary: str
    keywords: list[str]
    lang: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Union[SummarizeResponse, None] = None


# --- Authentication Endpoints ---

@auth_router.post("/register", response_model=schemas.User)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)

@auth_router.post("/login", response_model=schemas.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, email=form_data.username)
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@auth_router.get("/users/me", response_model=schemas.User)
def read_users_me(current_user: schemas.User = Depends(deps.get_current_user)):
    return current_user

# --- Finetune Endpoints ---

@finetune_router.post("/datasets/upload", response_model=schemas.FinetuneDataset)
async def upload_finetune_dataset(
    file: UploadFile = File(...),
    current_user: schemas.User = Depends(deps.get_current_user),
    db: Session = Depends(get_db)
):
    upload_dir = "finetune_datasets"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_location = os.path.join(upload_dir, f"{current_user.id}_{file.filename}")
    
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")

    db_dataset = crud.create_finetune_dataset(db, current_user.id, file_location)
    return db_dataset

@finetune_router.get("/datasets", response_model=List[schemas.FinetuneDataset])
def list_finetune_datasets(
    current_user: schemas.User = Depends(deps.get_current_user),
    db: Session = Depends(get_db)
):
    return crud.get_finetune_datasets(db, current_user.id)

@finetune_router.post("/models/train", response_model=schemas.FinetuneModel)
def train_finetune_model(
    model_name: str,
    base_model: str,
    dataset_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
    db: Session = Depends(get_db)
):
    # Check if dataset belongs to the user
    dataset = crud.get_finetune_datasets(db, current_user.id)
    if not any(d.id == dataset_id for d in dataset):
        raise HTTPException(status_code=404, detail="Dataset not found or does not belong to user")

    # Create a pending finetune model entry
    finetune_model_data = schemas.FinetuneModelCreate(
        model_name=model_name,
        base_model=base_model,
        status="pending",
        model_path=None
    )
    db_finetune_model = crud.create_finetune_model(db, current_user.id, finetune_model_data)

    # Enqueue the fine-tuning job
    queue.enqueue(
        finetuner.run_finetuning_job,
        current_user.id,
        dataset_id,
        model_name,
        base_model,
        db_finetune_model.id,
        # Pass db session to the worker to update status
        on_success=lambda job: update_finetune_model_status_in_worker(job, db_finetune_model.id, "completed"),
        on_failure=lambda job: update_finetune_model_status_in_worker(job, db_finetune_model.id, "failed")
    )
    
    return db_finetune_model

@finetune_router.get("/models", response_model=List[schemas.FinetuneModel])
def list_finetune_models(
    current_user: schemas.User = Depends(deps.get_current_user),
    db: Session = Depends(get_db)
):
    return crud.get_finetune_models(db, current_user.id)

# Helper function for RQ worker to update model status
def update_finetune_model_status_in_worker(job, model_id, status):
    db = next(get_db())
    model_path = job.result['model_path'] if job.result and 'model_path' in job.result else None
    crud.update_finetune_model_status(db, model_id, status, model_path)
    db.close()

# --- Webhook Endpoints ---
@webhook_router.post("/", response_model=schemas.Webhook)
def create_user_webhook(
    webhook: schemas.WebhookCreate,
    current_user: schemas.User = Depends(deps.get_current_user),
    db: Session = Depends(get_db)
):
    # Basic validation for event_type
    supported_events = ["summary_completed", "finetune_completed"]
    if webhook.event_type not in supported_events:
        raise HTTPException(status_code=400, detail=f"Unsupported event type. Supported types are: {', '.join(supported_events)}")

    db_webhook = crud.create_webhook(db, current_user.id, webhook)
    return db_webhook

@webhook_router.get("/", response_model=List[schemas.Webhook])
def list_user_webhooks(
    current_user: schemas.User = Depends(deps.get_current_user),
    db: Session = Depends(get_db)
):
    return crud.get_webhooks(db, current_user.id)

@webhook_router.delete("/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user_webhook(
    webhook_id: int,
    current_user: schemas.User = Depends(deps.get_current_user),
    db: Session = Depends(get_db)
):
    db_webhook = crud.get_webhook_by_id(db, webhook_id)
    if not db_webhook or db_webhook.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Webhook not found or does not belong to user")
    
    crud.delete_webhook(db, webhook_id)
    return

# --- Task Creation Endpoints ---

@router.post("/summarize", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
def summarize_text(
    request: SummarizeRequest,
    summary_length: SummaryLength = Query(SummaryLength.medium, alias="summary_length"),
    summary_type: SummaryType = Query(SummaryType.abstractive, alias="summary_type"),
    extractive_method: ExtractiveMethod = Query(ExtractiveMethod.lsa, alias="extractive_method"),
    tone: Tone = Query(Tone.neutral, alias="tone"),
    finetuned_model_id: Optional[int] = Query(None, alias="finetuned_model_id"), # New parameter
    current_user: schemas.User = Depends(deps.get_current_user),
):
    if not request.text:
        raise EmptyContentError("No text provided for summarization.")

    job = queue.enqueue(
        process_summary_task,
        request.text,
        summary_length.value,
        summary_type.value,
        extractive_method.value,
        tone.value,
        current_user.id,
        finetuned_model_id, # Pass new parameter
    )
    return {"task_id": job.id, "status": job.get_status()}

@router.post("/summarize_file", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def summarize_file(
    file: UploadFile = File(...),
    summary_length: SummaryLength = Query(SummaryLength.medium, alias="summary_length"),
    summary_type: SummaryType = Query(SummaryType.abstractive, alias="summary_type"),
    extractive_method: ExtractiveMethod = Query(ExtractiveMethod.lsa, alias="extractive_method"),
    finetuned_model_id: Optional[int] = Query(None, alias="finetuned_model_id"), # New parameter
    current_user: schemas.User = Depends(deps.get_current_user),
):
    file_extension = os.path.splitext(file.filename)[1].lower()
    supported_formats = [".txt", ".pdf", ".docx"]

    if file_extension not in supported_formats:
        raise UnsupportedFileTypeError(f"Unsupported file format: {file_extension}.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as e:
        raise FileProcessingError(f"Could not save uploaded file: {e}")

    text = ""
    try:
        if file_extension == ".txt":
            text = read_text_file(tmp_path)
        elif file_extension == ".pdf":
            text = read_pdf_text(tmp_path)
        elif file_extension == ".docx":
            text = read_docx_text(tmp_path)
    except Exception as e:
        raise FileProcessingError(f"Failed to read content from file: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not text:
        raise EmptyContentError("Could not read text from the file or file is empty.")

    job = queue.enqueue(
        process_summary_task,
        text,
        summary_length.value,
        summary_type.value,
        extractive_method.value,
        current_user.id,
        finetuned_model_id, # Pass new parameter
    )
    return {"task_id": job.id, "status": job.get_status()}


@router.post("/summarize_url", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
def summarize_url(
    request: UrlRequest,
    summary_length: SummaryLength = Query(SummaryLength.medium, alias="summary_length"),
    summary_type: SummaryType = Query(SummaryType.abstractive, alias="summary_type"),
    extractive_method: ExtractiveMethod = Query(ExtractiveMethod.lsa, alias="extractive_method"),
    finetuned_model_id: Optional[int] = Query(None, alias="finetuned_model_id"), # New parameter
    current_user: schemas.User = Depends(deps.get_current_user),
):
    try:
        text = get_text_from_url(request.url)
    except Exception as e:
        raise UrlConnectionError(str(e))

    if not text:
        raise EmptyContentError("Could not extract text from the URL.")

    job = queue.enqueue(
        process_summary_task,
        text,
        summary_length.value,
        summary_type.value,
        extractive_method.value,
        current_user.id,
        finetuned_model_id, # Pass new parameter
    )
    return {"task_id": job.id, "status": job.get_status()}

@router.post("/summarize_multiple", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
def summarize_multiple(
    request: schemas.MultiSummarizeRequest,
    summary_length: SummaryLength = Query(SummaryLength.medium, alias="summary_length"),
    summary_type: SummaryType = Query(SummaryType.abstractive, alias="summary_type"),
    extractive_method: ExtractiveMethod = Query(ExtractiveMethod.lsa, alias="extractive_method"),
    finetuned_model_id: Optional[int] = Query(None, alias="finetuned_model_id"),
    current_user: schemas.User = Depends(deps.get_current_user),
):
    if not request.texts:
        raise EmptyContentError("No texts provided for summarization.")
    
    job = queue.enqueue(
        process_multiple_summary_task, # New helper function for multiple documents
        request.texts,
        summary_length.value,
        summary_type.value,
        extractive_method.value,
        current_user.id,
        finetuned_model_id,
    )
    return {"task_id": job.id, "status": job.get_status()}


# --- Task Status Endpoint ---

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    job = Job.fetch(task_id, connection=redis_conn)
    
    response = {
        "task_id": job.id,
        "status": job.get_status(),
        "result": None
    }

    if job.is_finished:
        response["result"] = job.result
    elif job.is_failed:
        response["result"] = {"summary": job.exc_info, "keywords": [], "lang": ""}

    return response

# --- History Endpoint ---
@router.get("/summaries/history", response_model=List[schemas.SummaryHistory])
def get_summaries_history(
    current_user: schemas.User = Depends(deps.get_current_user),
    db: Session = Depends(get_db)
):
    return crud.create_summary_history(db=db, user=current_user, summary_data=gist)

@router.get("/usage", response_model=schemas.UsageStats)
def get_usage_stats(
    db: Session = Depends(deps.get_db),
    current_user: schemas.User = Depends(deps.get_current_user)
):
    summaries = crud.get_user_summaries(db, current_user)

    total_gists = len(summaries)

    # Gists in the last 30 days
    thirty_days_ago = datetime.now() - timedelta(days=30)
    gists_last_30_days = sum(1 for s in summaries if s.created_at >= thirty_days_ago)

    # Daily gists for the last 7 days
    daily_gists_last_7_days = {}
    for i in range(7):
        date = datetime.now() - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        daily_gists_last_7_days[date_str] = 0

    for s in summaries:
        if s.created_at >= datetime.now() - timedelta(days=7):
            date_str = s.created_at.strftime("%Y-%m-%d")
            if date_str in daily_gists_last_7_days:
                daily_gists_last_7_days[date_str] += 1

    return schemas.UsageStats(
        total_gists=total_gists,
        gists_last_30_days=gists_last_30_days,
        daily_gists_last_7_days=daily_gists_last_7_days
    )

# --- Other Endpoints ---

@router.get("/health")
async def health_check():
    return {"status": "ok"}

# --- Helper function to be called by RQ ---
def process_summary_task(text, length, summary_type, extractive_method, user_id, finetuned_model_id: Optional[int] = None):
    # Create a cache key
    cache_key = hashlib.md5(text.encode()).hexdigest()
    
    # Check cache first
    cached_result = cache.get_from_cache(cache_key)
    if cached_result:
        return cached_result

    # If not in cache, run summarization
    summary_result = summarize_long_text(text, length, summary_type, extractive_method, user_id, finetuned_model_id) # Pass new parameter
    
    # Extract entities from the original text
    entities = extract_entities(text, summary_result.get("lang", "en")) # Pass detected language
    summary_result["entities"] = entities # Add entities to the result

    # Save to cache
    cache.set_to_cache(cache_key, summary_result)

    # Save to history
    db = next(get_db())
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user:
        summary_data = schemas.SummaryHistoryCreate(
            input_text=text,
            summary=summary_result['summary'],
            title=text[:100], # Simple title for now
            model_used=finetuned_model_id if finetuned_model_id else "default", # Record model used
            entities=entities # Save entities to history
        )
        crud.create_summary_history(db, user, summary_data)

        # Dispatch webhooks
        webhooks = crud.get_webhooks(db, user_id)
        for webhook in webhooks:
            if webhook.event_type == "summary_completed":
                payload = {
                    "task_id": "N/A", # Task ID is not directly available here
                    "user_id": user_id,
                    "summary": summary_result['summary'],
                    "status": "completed",
                    "timestamp": datetime.now().isoformat()
                }
                dispatch_webhook(webhook.id, payload)
    
    return summary_result

def process_multiple_summary_task(texts, length, summary_type, extractive_method, user_id, finetuned_model_id: Optional[int] = None):
    # For multi-document summarization, caching is more complex (hash of all texts)
    # For simplicity, we'll skip caching for multi-document for now, or implement a more robust key generation.
    # cache_key = hashlib.md5(json.dumps(texts, sort_keys=True).encode()).hexdigest()
    # cached_result = cache.get_from_cache(cache_key)
    # if cached_result:
    #     return cached_result

    summary_result = summarize_multiple_documents(texts, user_id, length, summary_type, extractive_method, finetuned_model_id)

    # Extract entities from the combined summaries text (or from each original text and combine)
    # For simplicity, let's extract from the final summary for now.
    entities = extract_entities(summary_result.get("summary", ""), summary_result.get("lang", "en"))
    summary_result["entities"] = entities # Add entities to the result

    # Save to history (for multi-document, input_text might be too long, consider saving a reference or truncated version)
    db = next(get_db())
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user:
        # Truncate input_text for history if it's too long
        input_text_for_history = " ".join(texts)[:500] + "..." if len(" ".join(texts)) > 500 else " ".join(texts)
        summary_data = schemas.SummaryHistoryCreate(
            input_text=input_text_for_history,
            summary=summary_result['summary'],
            title=f"Multi-Doc Summary ({len(texts)} docs)", # Custom title for multi-doc
            model_used=finetuned_model_id if finetuned_model_id else "default",
            entities=entities # Save entities to history
        )
        crud.create_summary_history(db, user, summary_data)

        # Dispatch webhooks
        webhooks = crud.get_webhooks(db, user_id)
        for webhook in webhooks:
            if webhook.event_type == "summary_completed":
                payload = {
                    "task_id": "N/A", # Task ID is not directly available here
                    "user_id": user_id,
                    "summary": summary_result['summary'],
                    "status": "completed",
                    "timestamp": datetime.now().isoformat()
                }
                dispatch_webhook(webhook.id, payload)
    
    # cache.set_to_cache(cache_key, summary_result) # Re-enable if caching is implemented for multi-doc
    return summary_result

@router.websocket("/ws/summarize")
async def websocket_summarize(websocket: WebSocket):
    await websocket.accept()
    summarizer = StreamingSummarizer()
    try:
        while True:
            data = await websocket.receive_text()
            summarizer.add_chunk(data)
            summary = summarizer.get_summary()
            await websocket.send_text(summary)
    except WebSocketDisconnect:
        print("Client disconnected")
ait websocket.send_text(summary)
    except WebSocketDisconnect:
        print("Client disconnected")
