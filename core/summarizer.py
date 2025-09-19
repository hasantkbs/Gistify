from __future__ import annotations

from config.settings import ENGLISH_MODEL_NAME, MT5_MODEL_NAME, TURKISH_MODEL_NAME

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, DetectorFactory
import yake

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Import database and crud for fine-tuned model lookup
from api.database import get_db
from api import crud, models
from sqlalchemy.orm import Session


try:
    from .utils import LANG_CODE_MAP as _LANG_CODE_MAP  # type: ignore
except Exception:
    _LANG_CODE_MAP = {
        "tr": "turkish",
        "en": "english",
        "de": "german",
        "fr": "french",
        "es": "spanish",
        "it": "italian",
        "ar": "arabic",
        "ru": "russian",
    }

DetectorFactory.seed = 0

INPUT_CHARS_PER_CHUNK = 4000
SECOND_PASS_THRESHOLD = 5000

_PIPELINES: Dict[str, any] = {}

# ---------- Yardımcılar ----------

def _clean_summary(text: str) -> str:
    return " ".join((text or "").split())

def _preclean_text(text: str) -> str:
    import re
    if not text:
        return ""
    # Hyphenated line breaks: "nere-\nden" -> "nereden"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Single newlines -> space (keep paragraphs)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse spaces
    text = re.sub(r"\s+", " ", text)
    # Punctuation spacing
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(\S)", r"\1 \2", text)
    # Deduplicate repeated tokens / short phrases (PDF glitches)
    text = re.sub(r"\b(\w{2,})\b(?:\s*,?\s*\1\b){1,16}", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w{2,}\s+\w{2,})\b(?:\s*,?\s*\1\b){1,10}", r"\1", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _soft_sentence_split(text: str) -> List[str]:
    import re
    pieces = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in pieces if p]


def split_text_into_chunks(text: str, max_chars: int = INPUT_CHARS_PER_CHUNK) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    cur_len = 0
    for sent in _soft_sentence_split(text):
        s = sent if sent.endswith(" ") else sent + " "
        if cur_len + len(s) <= max_chars or not buf:
            buf.append(s)
            cur_len += len(s)
        else:
            chunks.append("".join(buf).strip())
            buf = [s]
            cur_len = len(s)
    if buf:
        chunks.append("".join(buf).strip())
    return chunks or [text]

@dataclass
class LengthSpec:
    name: str
    min_ratio: float
    max_ratio: float

_LENGTHS: Dict[str, LengthSpec] = {
    "short": LengthSpec("short", 0.12, 0.22),
    "medium": LengthSpec("medium", 0.22, 0.4),
    "long": LengthSpec("long", 0.4, 0.65),
}

def _pick_model_for_lang(lang_code: str, finetuned_model_id: Optional[int] = None) -> str:
    if finetuned_model_id:
        # In a real scenario, you would load the fine-tuned model path from the DB
        # and then load the model from that path.
        # For now, we'll simulate this by returning a special identifier.
        db = next(get_db()) # Get a new DB session for the worker
        finetuned_model = crud.get_finetune_model_by_id(db, finetuned_model_id)
        db.close()
        if finetuned_model and finetuned_model.model_path:
            print(f"Using fine-tuned model: {finetuned_model.model_name} from {finetuned_model.model_path}")
            return finetuned_model.model_path # Return the path to the fine-tuned model
        else:
            print(f"Fine-tuned model with ID {finetuned_model_id} not found or path is missing. Falling back to base model.")

    lang_code = (lang_code or "").lower()
    if lang_code.startswith("tr"):
        return TURKISH_MODEL_NAME
    if lang_code.startswith("en"):
        return ENGLISH_MODEL_NAME
    return MT5_MODEL_NAME


def _get_summarizer_pipeline(model_identifier: str):
    if model_identifier in _PIPELINES:
        return _PIPELINES[model_identifier]

    print(f"Loading model: {model_identifier}")
    if "t5" in model_identifier.lower() or "mt5" in model_identifier.lower(): # Check for mt5 as well
        tok = AutoTokenizer.from_pretrained(model_identifier, use_fast=False, legacy=False, model_max_length=512)
        print(f"  model_max_length: 512 (T5/mT5 model)")
    else:
        tok = AutoTokenizer.from_pretrained(model_identifier, model_max_length=1024)
        print(f"  model_max_length: 1024 (Non-T5 model)")
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_identifier)

    _PIPELINES[model_identifier] = pipeline("summarization", model=mdl, tokenizer=tok)
    return _PIPELINES[model_identifier]


def _ratio_to_lengths(text: str, length_key: str) -> Tuple[int, int]:
    spec = _LENGTHS.get(length_key, _LENGTHS["medium"])
    word_count = max(1, len((text or "").split()))
    min_len = max(30, int(word_count * spec.min_ratio))
    max_len = max(min_len + 5, int(word_count * spec.max_ratio))
    return (min_len, max_len)


def summarize_with_hf(text: str, detected_lang: str, length_key: str, finetuned_model_id: Optional[int] = None) -> str:
    print(f"Summarizing text (length: {len(text)}) with HF model for lang: {detected_lang}")
    model_identifier = _pick_model_for_lang(detected_lang, finetuned_model_id) # Pass finetuned_model_id
    summarizer = _get_summarizer_pipeline(model_identifier)
    min_len, max_len = _ratio_to_lengths(text, length_key)
    out = summarizer(
        text,
        min_length=min_len,
        max_length=max_len,
        do_sample=False,
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        truncation=True,
    )
    if isinstance(out, list) and out and isinstance(out[0], dict) and "summary_text" in out[0]:
        return out[0]["summary_text"].strip()
    if isinstance(out, dict) and "summary_text" in out:
        return out["summary_text"].strip()
    return text


def summarize_with_sumy(text: str, method: str = "textrank", sentence_count: int = 5) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    if method == "lsa":
        summarizer = LsaSummarizer()
    elif method == "lexrank":
        summarizer = LexRankSummarizer()
    else:
        summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, sentences_count=sentence_count)
    return " ".join(str(s) for s in sentences).strip() or text


def extract_keywords(text: str, lang_code: str | None = None, top_k: int = 8) -> List[str]:
    lang_code = (lang_code or "en").lower()
    lang = _LANG_CODE_MAP.get(lang_code[:2], "english")
    kw = yake.KeywordExtractor(lan=lang, top=top_k, n=1)
    items = kw.extract_keywords(text)
    return [k for k, _ in sorted([(t[0], t[1]) for t in items], key=lambda x: x[0])]

# ---------- Ana API ----------

def summarize_text(text: str, length: str = "medium", summary_type: str = "abstractive", extractive_method: str = "textrank", user_id: str | None = None, finetuned_model_id: Optional[int] = None) -> dict:
    text = _preclean_text(text or "")
    if not text:
        return {"summary": "", "keywords": [], "lang": ""}

    try:
        detected_lang = detect(text)
    except Exception:
        detected_lang = "en"

    def _sent_count_for_len(len_key: str) -> int:
        if len_key == "short":
            return 4
        if len_key == "long":
            return 10
        return 6

    summary_out = ""
    if summary_type.lower() == "extractive":
        method = extractive_method.lower()
        if method not in {"lsa", "lexrank", "textrank"}:
            method = "textrank"
        summary_out = summarize_with_sumy(text, method=method, sentence_count=_sent_count_for_len(length))
    else:
        # Improved Hybrid Approach
        # 1. Get the extractive summary to serve as the context/skeleton.
        # Use a slightly larger sentence count for the extractive step to provide more context.
        extractive_sentence_count = max(8, _sent_count_for_len(length) * 2)
        extractive_summary = summarize_with_sumy(text, method="lexrank", sentence_count=extractive_sentence_count)

        # 2. Use the extractive summary as the input for the abstractive model.
        # This helps the model focus on the most important parts of the text.
        if len(extractive_summary) > INPUT_CHARS_PER_CHUNK:
             # If the extractive summary is still too long, chunk it.
             chunks = split_text_into_chunks(extractive_summary, INPUT_CHARS_PER_CHUNK)
             partials: List[str] = []
             for ch in chunks:
                 try:
                     s = summarize_with_hf(ch, detected_lang, length, finetuned_model_id) # Pass finetuned_model_id
                 except Exception:
                     s = summarize_with_sumy(ch, method="textrank", sentence_count=5)
                 partials.append(_clean_summary(s))
             combined = "\n\n".join(partials).strip()
             if len(combined) > SECOND_PASS_THRESHOLD:
                 try:
                     combined = summarize_with_hf(combined, detected_lang, length, finetuned_model_id) # Pass finetuned_model_id
                 except Exception:
                     combined = summarize_with_sumy(combined, method="lexrank", sentence_count=7)
                 combined = _clean_summary(combined)
             summary_out = combined
        else:
             summary_out = summarize_with_hf(extractive_summary, detected_lang, length, finetuned_model_id) # Pass finetuned_model_id


    try:
        kws = extract_keywords(text, detected_lang)
    except Exception:
        kws = []

    return {"summary": _clean_summary(summary_out), "keywords": kws, "lang": detected_lang}


def summarize_long_text(text: str, user_id: str | None = None, length: str = "medium", summary_type: str = "abstractive", extractive_method: str = "textrank", finetuned_model_id: Optional[int] = None) -> dict:
    return summarize_text(text, length=length, summary_type=summary_type, extractive_method=extractive_method, user_id=user_id, finetuned_model_id=finetuned_model_id)

def summarize_multiple_documents(
    texts: List[str],
    user_id: str | None = None,
    length: str = "medium",
    summary_type: str = "abstractive",
    extractive_method: str = "textrank",
    finetuned_model_id: Optional[int] = None
) -> dict:
    """
    Summarizes multiple documents using a two-stage approach.
    First, summarizes each document individually, then summarizes the combined individual summaries.
    """
    if not texts:
        return {"summary": "", "keywords": [], "lang": ""}

    individual_summaries = []
    for i, text in enumerate(texts):
        print(f"Summarizing document {i+1}/{len(texts)}...")
        # Use summarize_long_text for individual document summarization
        # For individual summaries, we might want a 'long' length to retain more info
        # or a specific length based on the overall desired output.
        # For simplicity, let's use 'medium' for individual summaries for now.
        individual_summary_result = summarize_long_text(
            text,
            user_id=user_id,
            length="medium", # Summarize each document to a medium length
            summary_type=summary_type,
            extractive_method=extractive_method,
            finetuned_model_id=finetuned_model_id
        )
        if individual_summary_result and individual_summary_result.get("summary"):
            individual_summaries.append(individual_summary_result["summary"])

    if not individual_summaries:
        return {"summary": "Could not summarize any of the provided documents.", "keywords": [], "lang": ""}

    # Combine individual summaries
    combined_summaries_text = "\n\n".join(individual_summaries)
    print(f"Combined {len(individual_summaries)} individual summaries. Total length: {len(combined_summaries_text)}")

    # Summarize the combined summaries
    final_summary_result = summarize_long_text(
        combined_summaries_text,
        user_id=user_id,
        length=length, # Use the requested final length
        summary_type=summary_type,
        extractive_method=extractive_method,
        finetuned_model_id=finetuned_model_id
    )
    
    # Combine keywords from individual summaries (simple approach)
    all_keywords = []
    for i, text in enumerate(texts):
        try:
            detected_lang = detect(text)
        except Exception:
            detected_lang = "en"
        kws = extract_keywords(text, detected_lang)
        all_keywords.extend(kws)
    
    # Deduplicate and return top keywords
    final_keywords = list(dict.fromkeys(all_keywords))[:10] # Get unique and top 10

    return {
        "summary": final_summary_result.get("summary", "No final summary generated."),
        "keywords": final_keywords,
        "lang": final_summary_result.get("lang", "en") # Use lang from final summary or default
    }
