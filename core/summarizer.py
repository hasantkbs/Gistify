from transformers import pipeline
from langdetect import detect, DetectorFactory
import yake
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lexrank import LexRankSummarizer
from sumy.summarizers.textrank import TextRankSummarizer
from .utils import LANG_CODE_MAP, MAX_CHUNK_CHARS, clean_summary, split_text_into_chunks
from config.settings import BART_MODEL_NAME, MT5_MODEL_NAME, TURKISH_MODEL_NAME

# Set seed for reproducibility in langdetect
DetectorFactory.seed = 0

# Store loaded pipelines to avoid reloading
_summarizer_pipelines = {}

def get_summarizer_pipeline(lang_code: str):
    global _summarizer_pipelines

    if lang_code == "en":
        model_name = BART_MODEL_NAME
    elif lang_code == "tr":
        model_name = TURKISH_MODEL_NAME
    else:
        model_name = MT5_MODEL_NAME

    if model_name not in _summarizer_pipelines:
        print(f"Loading summarization model: {model_name}...")
        _summarizer_pipelines[model_name] = pipeline("summarization", model=model_name)
        print(f"Model {model_name} loaded.")
    return _summarizer_pipelines[model_name]

def summarize_chunk(text: str, length: str = "medium") -> str:
    """
    Verilen metin parçasını yerel özetleyici model ile özetler.
    """
    try:
        # Detect language
        detected_lang = "en" # Default to English if detection fails
        try:
            detected_lang = detect(text)
        except:
            pass # langdetect can fail on very short or non-text inputs

        # Get the appropriate summarizer pipeline based on detected language
        summarizer_pipeline = get_summarizer_pipeline(detected_lang)

        # Define length ratios based on the length parameter
        length_ratios = {
            "short": (0.15, 0.30),
            "medium": (0.30, 0.60),
            "long": (0.50, 0.80)
        }
        min_ratio, max_ratio = length_ratios.get(length, (0.30, 0.60))

        # Calculate dynamic min_length and max_length
        words = len(text.split())
        min_length = max(10, int(words * min_ratio))
        max_length = min(300, int(words * max_ratio) + 20)

        # Ensure max_length is not less than min_length
        if max_length < min_length:
            max_length = min_length + 15

        summary_list = summarizer_pipeline(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            num_beams=4 # Consistent summary
        )
        return summary_list[0]['summary_text'].strip()
    except Exception as e:
        return f"Özetleme sırasında bir hata oluştu: {e}"

def extract_keywords(text: str, lang_code: str = "en") -> list[str]:
    """
    Extracts keywords from the given text using YAKE.
    """
    try:
        # YAKE language codes are typically the first two letters (e.g., 'en', 'tr')
        language = lang_code[:2]
        kw_extractor = yake.KeywordExtractor(lan=language, top=5, n=2)
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]
    except Exception as e:
        print(f"Keyword extraction failed: {e}")
        return []

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'punkt' model...")
        nltk.download('punkt')
        print("NLTK 'punkt' model downloaded.")

# Call this on module load to ensure 'punkt' is available
download_nltk_data()

def summarize_lexrank(text: str, lang: str = "english", sentences_count: int = 5) -> str:
    """
    Generates an extractive summary using Sumy (LexRank).
    """
    try:
        parser = PlaintextParser.from_string(text, Tokenizer(lang))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, sentences_count)
        return " ".join([str(sentence) for sentence in summary])
    except Exception as e:
        return f"Extractive summarization (LexRank) failed: {e}"

def summarize_textrank(text: str, lang: str = "english", sentences_count: int = 5) -> str:
    """
    Generates an extractive summary using Sumy (TextRank).
    """
    try:
        parser = PlaintextParser.from_string(text, Tokenizer(lang))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, sentences_count)
        return " ".join([str(sentence) for sentence in summary])
    except Exception as e:
        return f"Extractive summarization (TextRank) failed: {e}"

def summarize_extractive(text: str, lang: str = "english", sentences_count: int = 5) -> str:
    """
    Generates an extractive summary using Sumy (LSA).
    """
    try:
        parser = PlaintextParser.from_string(text, Tokenizer(lang))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentences_count)
        return " ".join([str(sentence) for sentence in summary])
    except Exception as e:
        return f"Extractive summarization failed: {e}"

def summarize_long_text(long_text: str, user_id: str = None, length: str = "medium", summary_type: str = "abstractive", extractive_method: str = "lsa") -> dict:
    """
    Summarizes long texts and extracts keywords.
    Returns a dictionary with 'summary' and 'keywords'.
    """
    # Detect language for keyword extraction and extractive summarization
    detected_lang = "en"
    try:
        detected_lang = detect(long_text)
    except:
        pass

    # Extract keywords from the original full text
    keywords = extract_keywords(long_text, lang_code=detected_lang)

    # --- Summarization Process ---
    if summary_type == "extractive":
        # Determine number of sentences based on length
        length_map = {"short": 3, "medium": 5, "long": 8}
        num_sentences = length_map.get(length, 5)
        
        # Map langdetect code to sumy language name
        sumy_lang_map = {"en": "english", "tr": "turkish"} # Add more mappings as needed
        sumy_lang = sumy_lang_map.get(detected_lang, "english")

        if extractive_method == "lexrank":
            summary_text = summarize_lexrank(long_text, lang=sumy_lang, sentences_count=num_sentences)
        elif extractive_method == "textrank":
            summary_text = summarize_textrank(long_text, lang=sumy_lang, sentences_count=num_sentences)
        else: # Default to LSA
            summary_text = summarize_extractive(long_text, lang=sumy_lang, sentences_count=num_sentences)
    else: # Default to abstractive
        chunks = split_text_into_chunks(long_text)
        if not chunks:
            return {"summary": "Could not find text to summarize.", "keywords": keywords}

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)}...")
            summary = summarize_chunk(chunk, length=length)
            if "hata" in summary.lower():
                return {"summary": summary, "keywords": keywords} # Return error with keywords
            chunk_summaries.append(summary)
        
        combined_summary = "\n\n".join(chunk_summaries)
        
        if len(combined_summary) > MAX_CHUNK_CHARS * 1.5:
            print("Combined summary is too long, creating a summary of summaries...")
            summary_text = summarize_chunk(combined_summary, length="medium")
        else:
            summary_text = combined_summary

    cleaned_summary = clean_summary(summary_text)

    return {"summary": cleaned_summary, "keywords": keywords}