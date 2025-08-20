from fastapi import APIRouter
from pydantic import BaseModel
import os

from core.summarizer import summarize_long_text
from input_handlers.pdf_handler import read_pdf_text
from input_handlers.docx_handler import read_docx_text
from input_handlers.text_handler import read_text_file
from core.utils import clean_summary # Assuming clean_summary is in core.utils

router = APIRouter()

class SummarizeRequest(BaseModel):
    text: str

class SummarizeFileRequest(BaseModel):
    file_path: str

@router.post("/summarize")
async def summarize_text(request: SummarizeRequest):
    """
    Summarizes the provided text.
    """
    text = request.text
    if not text:
        return {"error": "No text provided for summarization."}
    
    summary = summarize_long_text(text)
    if summary.startswith("Özetleme sırasında bir hata oluştu:") or \
       summary.startswith("API isteği sırasında bir hata oluştu:") or \
       summary.startswith("API'den beklenmedik yanıt yapısı:") or \
       summary.startswith("Yapılandırma hatası:") or \
       summary.startswith("Özetleme sırasında beklenmeyen bir hata oluştu:") :
        return {"error": summary}
    
    cleaned_summary = clean_summary(summary)
    return {"summary": cleaned_summary}

@router.post("/summarize_file")
async def summarize_file(request: SummarizeFileRequest):
    """
    Summarizes the content of a file.
    Supports .txt, .pdf, and .docx files.
    """
    file_path = request.file_path
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    file_extension = os.path.splitext(file_path)[1].lower()
    text = ""
    if file_extension == ".txt":
        text = read_text_file(file_path)
        if text.startswith("Metin dosyası okuma hatası:"):
            return {"error": text}
    elif file_extension == ".pdf":
        text = read_pdf_text(file_path)
        if text.startswith("PDF okuma hatası:"):
            return {"error": text}
    elif file_extension == ".docx":
        text = read_docx_text(file_path)
        if text.startswith("DOCX okuma hatası:"):
            return {"error": text}
    else:
        return {"error": f"Unsupported file format: {file_extension}. Only .txt, .pdf, and .docx are supported."}

    if not text:
        return {"error": "Could not read text from the file or file is empty."}

    summary = summarize_long_text(text)
    if summary.startswith("Özetleme sırasında bir hata oluştu:") or \
       summary.startswith("API isteği sırasında bir hata oluştu:") or \
       summary.startswith("API'den beklenmedik yanıt yapısı:") or \
       summary.startswith("Yapılandırma hatası:") or \
       summary.startswith("Özetleme sırasında beklenmeyen bir hata oluştu:") :
        return {"error": summary}
    
    cleaned_summary = clean_summary(summary)
    return {"summary": cleaned_summary}
