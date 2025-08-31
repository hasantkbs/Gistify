from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query, status
from pydantic import BaseModel
import os
import tempfile
from enum import Enum
from .deps import get_current_user
from .crud import update_user_subscription, increment_summary_count
from .models import UserSubscription

from core.summarizer import summarize_long_text
from input_handlers.pdf_handler import read_pdf_text
from input_handlers.docx_handler import read_docx_text
from input_handlers.text_handler import read_text_file
from core.utils import clean_summary

router = APIRouter()

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

class SummarizeResponse(BaseModel):
    summary: str
    keywords: list[str]

class SummarizeRequest(BaseModel):
    text: str

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(
    request: SummarizeRequest, 
    summary_length: SummaryLength = Query("medium", description="The desired length of the summary."),
    summary_type: SummaryType = Query("abstractive", description="The type of summarization to perform."),
    extractive_method: ExtractiveMethod = Query("lsa", description="The extractive summarization method to use."),
    user: dict = Depends(get_current_user)
):
    """
    Summarizes the provided text.
    """
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="No text provided for summarization.")
    
    # # Subscription check
    # if not user['subscription']['is_subscribed'] and user['subscription']['summaries_this_month'] >= 5:
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="Free users are limited to 5 summaries per month. Please subscribe to remove this limit."
    #     )

    result = summarize_long_text(text, length=summary_length.value)
    if "hata" in result["summary"].lower() or "failed" in result["summary"].lower():
        raise HTTPException(status_code=500, detail=result["summary"])
    
    # # Increment summary count
    # await increment_summary_count(user['uid'])

    return result

@router.post("/summarize_file", response_model=SummarizeResponse)
async def summarize_file(
    file: UploadFile = File(...), 
    summary_length: SummaryLength = Query("medium", description="The desired length of the summary."),
    summary_type: SummaryType = Query("abstractive", description="The type of summarization to perform."),
    extractive_method: ExtractiveMethod = Query("lsa", description="The extractive summarization method to use."),
    user: dict = Depends(get_current_user)
):
    """
    Summarizes the content of an uploaded file.
    Supports .txt, .pdf, and .docx files.
    """
    file_extension = os.path.splitext(file.filename)[1].lower()
    supported_formats = [".txt", ".pdf", ".docx"]

    if file_extension not in supported_formats:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}. Only .txt, .pdf, and .docx are supported.")

    # Save uploaded file to a temporary file to be read by handlers
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(await file.read())
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
    except Exception as e:
        # Re-raise exceptions from file readers with more context.
        raise HTTPException(status_code=500, detail=f"Error reading file content: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path) # Clean up the temporary file

    if not text:
        raise HTTPException(status_code=400, detail="Could not read text from the file or file is empty.")

    # # Subscription check
    # if not user['subscription']['is_subscribed'] and user['subscription']['summaries_this_month'] >= 5:
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="Free users are limited to 5 summaries per month. Please subscribe to remove this limit."
    #     )

    result = summarize_long_text(text, length=summary_length.value)
    if "hata" in result["summary"].lower() or "failed" in result["summary"].lower():
        raise HTTPException(status_code=500, detail=result["summary"])
    
    # # Increment summary count
    # await increment_summary_count(user['uid'])

    return result

@router.get("/users/me", response_model=UserSubscription)
async def read_users_me(user: dict = Depends(get_current_user)):
    """
    Get current user's subscription status and usage.
    """
    return UserSubscription(**user['subscription'])

@router.post("/subscribe")
async def subscribe_user(user: dict = Depends(get_current_user)):
    """
    Mock endpoint to simulate user subscription.
    Sets is_subscribed to True for the current user.
    """
    uid = user['uid']
    await update_user_subscription(uid, {"is_subscribed": True})
    return {"message": "Subscription successful (mock)."}

@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify that the service is running.
    """
    return {"status": "ok"}
