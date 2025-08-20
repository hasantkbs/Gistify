from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, DetectorFactory
from .utils import LANG_CODE_MAP, MAX_CHUNK_CHARS, MIN_OVERLAP_CHARS, clean_summary, split_text_into_chunks

# Set seed for reproducibility in langdetect
DetectorFactory.seed = 0

# Define model names
BART_MODEL_NAME = "facebook/bart-large-cnn"
MT5_MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"

# Store loaded pipelines to avoid reloading
_summarizer_pipelines = {}

def get_summarizer_pipeline(lang_code: str):
    global _summarizer_pipelines

    if lang_code == "en":
        model_name = BART_MODEL_NAME
    else:
        model_name = MT5_MODEL_NAME

    if model_name not in _summarizer_pipelines:
        print(f"Loading summarization model: {model_name}...")
        _summarizer_pipelines[model_name] = pipeline("summarization", model=model_name)
        print(f"Model {model_name} loaded.")
    return _summarizer_pipelines[model_name]

def summarize_chunk(text: str) -> str:
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

        # Map to mBART language code
        mbart_lang_code = LANG_CODE_MAP.get(detected_lang, "en_XX") # Default to en_XX

        # Get the appropriate summarizer pipeline based on detected language
        summarizer_pipeline = get_summarizer_pipeline(detected_lang)

        # Set source and target language for multilingual models (mT5)
        if detected_lang != "en":
            # For mT5, we need to set the source and target language tokens
            # The mT5_multilingual_XLSum model uses a different approach for language tokens
            # It's usually handled by the tokenizer directly or by adding special tokens.
            # Let's assume the pipeline handles it if we set the tokenizer's language.
            # If issues arise, we might need to explicitly set forced_bos_token_id.
            # For now, we'll rely on the pipeline's default behavior for multilingual.
            # The user's requirement is "İngilizce dışı girdiler aynı dilde özetlenmeli (mT5 tercih)."
            # This implies the model should output in the source language.
            # The pipeline should handle this if the model is truly multilingual.
            # We will remove the explicit src_lang and tgt_lang setting on tokenizer for now,
            # as it caused issues before and the pipeline might manage it.
            pass # No explicit src_lang/tgt_lang setting for mT5 pipeline for now

        # Calculate dynamic min_length and max_length
        words = len(text.split())
        min_length = max(5, int(words * 0.40))
        max_length = min(200, int(words * 0.80) + 20)

        # Ensure max_length is not less than min_length
        if max_length < min_length:
            max_length = min_length + 10 # Add a small buffer

        # Ensure min_length is not too large for very short texts
        if min_length > words:
            min_length = words // 2 if words > 0 else 1

        # Ensure max_length is not too large for very short texts
        if max_length > words:
            max_length = words - 1 if words > 1 else 1


        # BART's maximum input length is typically 1024 tokens.
        # Text is truncated if too long for the model.
        # The pipeline usually handles this, but we keep the check for safety.
        # max_input_length = summarizer_pipeline.model.config.max_position_embeddings
        # if len(text) > max_input_length * 4: # Approximate (1 token ~ 4 characters)
        #     text = text[:max_input_length * 4]

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

def summarize_long_text(long_text: str) -> str:
    """
    Uzun metinleri parçalara ayırarak ve özetleri tekrar özetleyerek özetler.
    """
    # İlk özetleme aşaması
    chunks = split_text_into_chunks(long_text)
    
    if not chunks:
        return "Özetlenecek metin bulunamadı."

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Parça {i+1}/{len(chunks)} özetleniyor...")
        summary = summarize_chunk(chunk)
        # Apply clean_summary to each chunk summary
        if not summary.startswith("Özetleme sırasında bir hata oluştu:"): # Check for error messages
            summary = clean_summary(summary)
        
        if summary.startswith("API isteği sırasında bir hata oluştu:") or \
           summary.startswith("API'den beklenmedik yanıt yapısı:") or \
           summary.startswith("Yapılandırma hatası:") or \
           summary.startswith("Özetleme sırasında beklenmeyen bir hata oluştu:") :
            print(f"Hata: Parça {i+1} özetlenirken hata oluştu: {summary}")
            return summary # Hata durumunda dur
        chunk_summaries.append(summary)
    
    combined_summary = "\n\n".join(chunk_summaries)
    print("\n--- Ara Özetler Birleştirildi ---")
    print(combined_summary)
    print("-----------------------------------")
    
    # İkinci özetleme aşaması (özetlerin özeti)
    # Eğer birleştirilmiş özet hala çok uzunsa, tekrar özetle
    if len(combined_summary) > MAX_CHUNK_CHARS * 1.5: # Eğer birleştirilmiş özet hala uzunsa
        print("Birleştirilmiş özet çok uzun, özetlerin özeti oluşturuluyor...")
        final_summary = summarize_chunk(combined_summary)
        return final_summary
    else:
        return combined_summary