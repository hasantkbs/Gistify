from __future__ import annotations

# pick a real summarization model for TR; BART for EN; mT5 otherwise
BART_MODEL_NAME = "facebook/bart-large-cnn"
TURKISH_MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"  # TR summarization
MT5_MODEL_NAME = "google/mt5-small"

from typing import Dict, List, Tuple
from dataclasses import dataclass

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, DetectorFactory
import yake

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer



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

def _pick_model_for_lang(lang_code: str) -> str:
    lang_code = (lang_code or "").lower()
    if lang_code.startswith("tr"):
        return TURKISH_MODEL_NAME
    if lang_code.startswith("en"):
        return BART_MODEL_NAME
    return MT5_MODEL_NAME


def _get_summarizer_pipeline(lang_code: str):
    model_name = _pick_model_for_lang(lang_code)
    if model_name in _PIPELINES:
        return _PIPELINES[model_name]

    # T5/mT5: avoid fast tokenizer UNKs and use new behavior
    if "t5" in model_name.lower():
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False, model_max_length=512)
    else:
        tok = AutoTokenizer.from_pretrained(model_name, model_max_length=1024)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    _PIPELINES[model_name] = pipeline("summarization", model=mdl, tokenizer=tok)
    return _PIPELINES[model_name]


def _ratio_to_lengths(text: str, length_key: str) -> Tuple[int, int]:
    spec = _LENGTHS.get(length_key, _LENGTHS["medium"])
    word_count = max(1, len((text or "").split()))
    min_len = max(30, int(word_count * spec.min_ratio))
    max_len = max(min_len + 5, int(word_count * spec.max_ratio))
    return (min_len, max_len)


def summarize_with_hf(text: str, detected_lang: str, length_key: str) -> str:
    summarizer = _get_summarizer_pipeline(detected_lang)
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

def summarize_text(text: str, length: str = "medium", summary_type: str = "abstractive", extractive_method: str = "textrank", user_id: str | None = None) -> dict:
    text = _preclean_text(text or "")
    if not text:
        return {"summary": "", "keywords": [], "lang": ""}

    try:
        detected_lang = detect(text)
    except Exception:
        detected_lang = "en"

    def _sent_count_for_len(len_key: str) -> int:
        return 4 if len_key == "short" else (10 if len_key == "long" else 6)

    if summary_type.lower() == "extractive":
        method = extractive_method.lower()
        if method not in {"lsa", "lexrank", "textrank"}:
            method = "textrank"
        summary_out = summarize_with_sumy(text, method=method, sentence_count=_sent_count_for_len(length))
    else:
        base_text = text
        if summary_type.lower() == "hybrid":
            method = extractive_method.lower() if extractive_method else "textrank"
            if method not in {"lsa", "lexrank", "textrank"}:
                method = "textrank"
            base_text = summarize_with_sumy(text, method=method, sentence_count=max(8, _sent_count_for_len(length) * 2))

        chunks = split_text_into_chunks(base_text, INPUT_CHARS_PER_CHUNK)
        partials: List[str] = []
        for ch in chunks:
            try:
                s = summarize_with_hf(ch, detected_lang, length)
            except Exception:
                s = summarize_with_sumy(ch, method="textrank", sentence_count=5)
            partials.append(_clean_summary(s))
        combined = "\n\n".join(partials).strip()
        if len(combined) > SECOND_PASS_THRESHOLD:
            try:
                combined = summarize_with_hf(combined, detected_lang, length)
            except Exception:
                combined = summarize_with_sumy(combined, method="lexrank", sentence_count=7)
            combined = _clean_summary(combined)
        summary_out = combined

    try:
        kws = extract_keywords(text, detected_lang)
    except Exception:
        kws = []

    return {"summary": _clean_summary(summary_out), "keywords": kws, "lang": detected_lang}


def summarize_long_text(text: str, user_id: str | None = None, length: str = "medium", summary_type: str = "abstractive", extractive_method: str = "textrank") -> dict:
    return summarize_text(text, length=length, summary_type=summary_type, extractive_method=extractive_method, user_id=user_id)
