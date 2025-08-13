#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
note_summarizer.py
Terminal üzerinden girilen veya dosyadan okunan notları
kısa, öz ve anlamlı bir şekilde özetler.
"""

import argparse
import os
from transformers import pipeline
from langdetect import detect, DetectorFactory

# Set seed for reproducibility in langdetect
DetectorFactory.seed = 0

# Summarization pipeline
# Model ilk çalıştırmada indirilecektir.
summarizer = pipeline("summarization", model="facebook/mbart-large-50-many-to-many-mmt")

# Mapping from langdetect codes to mBART-50 language codes
LANG_CODE_MAP = {
    "en": "en_XX",
    "tr": "tr_TR",
    "fr": "fr_XX",
    "de": "de_DE",
    "es": "es_XX",
    "it": "it_IT",
    "ru": "ru_RU",
    "ar": "ar_AR",
    "zh-cn": "zh_CN", # Simplified Chinese
    "pt": "pt_XX",
    "nl": "nl_XX",
    "ja": "ja_XX",
    "ko": "ko_KR",
    "hi": "hi_IN",
    "ur": "ur_PK",
    "fa": "fa_IR",
    "bn": "bn_IN",
    "vi": "vi_VN",
    "th": "th_TH",
    "id": "id_ID",
    "ms": "ms_MY",
    "sw": "sw_KE",
    "ha": "ha_NG",
    "pl": "pl_PL",
    "uk": "uk_UA",
    "ro": "ro_RO",
    "cs": "cs_CZ",
    "hu": "hu_HU",
    "fi": "fi_FI",
    "sv": "sv_SE",
    "da": "da_DK",
    "no": "no_NO",
    "el": "el_GR",
    "bg": "bg_BG",
    "sr": "sr_RS",
    "sk": "sk_SK",
    "sl": "sl_SI",
    "et": "et_EE",
    "lv": "lv_LV",
    "lt": "lt_LT",
    "hr": "hr_HR",
    "ca": "ca_ES",
    "eu": "eu_ES",
    "gl": "gl_ES",
    "af": "af_ZA",
    "am": "am_ET",
    "az": "az_AZ",
    "be": "be_BY",
    "gu": "gu_IN",
    "is": "is_IS",
    "ka": "ka_GE",
    "km": "km_KH",
    "lo": "lo_LA",
    "mk": "mk_MK",
    "ml": "ml_IN",
    "mn": "mn_MN",
    "my": "my_MM",
    "ne": "ne_NP",
    "om": "om_ET",
    "ps": "ps_AF",
    "so": "so_SO",
    "sq": "sq_AL",
    "ta": "ta_IN",
    "te": "te_IN",
    "ti": "ti_ET",
    "ug": "ug_CN",
    "uz": "uz_UZ",
    "xh": "xh_ZA",
    "yi": "yi_US",
    "yo": "yo_NG",
    "zu": "zu_ZA",
}


import re
import PyPDF2
from docx import Document

def read_pdf_text(file_path):
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() + "\n"
    except Exception as e:
        return f"PDF okuma hatası: {e}"
    return text

def read_docx_text(file_path):
    text = ""
    try:
        document = Document(file_path)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        return f"DOCX okuma hatası: {e}"
    return text

# Function to clean summary
def clean_summary(summary_text):
    # Remove common English adverbs that might be hallucinated
    summary_text = re.sub(r'\b(guiltful|grotesque|enthusiastically|worrisome|tahminiably)\b', '', summary_text, flags=re.IGNORECASE)
    # Remove non-Turkish characters (keep Turkish alphabet, numbers, punctuation, and spaces)
    # This is a more aggressive filter and might remove legitimate non-Turkish names/terms
    summary_text = re.sub(r"[^a-zA-Z0-9ğĞüÜşŞıİöÖçÇ\s.,;!?'\"()[\]-]", '', summary_text)
    # Clean up multiple spaces
    summary_text = re.sub(r'\s+', ' ', summary_text).strip()
    return summary_text

# Constants for chunking
MAX_CHUNK_CHARS = 3000 # Approximate character limit for a chunk
MIN_OVERLAP_CHARS = 200 # Overlap between chunks to maintain context



def split_text_into_chunks(text, max_chars=MAX_CHUNK_CHARS, overlap_chars=MIN_OVERLAP_CHARS):
    """Splits text into chunks, trying to respect paragraph boundaries."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > max_chars and current_chunk: # +2 for newline
            chunks.append(current_chunk.strip())
            current_chunk = para # Start new chunk with current paragraph
        else:
            current_chunk += (para + '\n\n')

    if current_chunk:
        chunks.append(current_chunk.strip())

    # If any chunk is still too large, split by sentences
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            sub_chunk = ""
            for sent in sentences:
                if len(sub_chunk) + len(sent) + 1 > max_chars and sub_chunk: 
                    final_chunks.append(sub_chunk.strip())
                    sub_chunk = sent
                else:
                    sub_chunk += (sent + ' ')
            if sub_chunk:
                final_chunks.append(sub_chunk.strip())
        else:
            final_chunks.append(chunk)
            
    # Add overlap for better context
    overlapped_chunks = []
    for i, chunk in enumerate(final_chunks):
        if i > 0:
            # Take a portion of the previous chunk to overlap
            prev_chunk_end = final_chunks[i-1][-overlap_chars:] if len(final_chunks[i-1]) > overlap_chars else final_chunks[i-1]
            overlapped_chunks.append(prev_chunk_end + "\n\n" + chunk)
        else:
            overlapped_chunks.append(chunk)

    return overlapped_chunks

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

        # Set the source language for the tokenizer
        summarizer.tokenizer.src_lang = mbart_lang_code

        # max_length ve min_length özetin uzunluğunu kontrol eder.
        # do_sample=False deterministik sonuçlar için.
        # BART'ın maksimum giriş uzunluğu genellikle 1024 token'dır.
        # Metin çok uzunsa, modelin kaldırabileceği boyuta kesilir.
        max_input_length = summarizer.model.config.max_position_embeddings
        if len(text) > max_input_length * 4: # Yaklaşık bir tahmin (1 token ~ 4 karakter)
            text = text[:max_input_length * 4]

        summary_list = summarizer(text, max_length=500, min_length=30, do_sample=False)
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
           summary.startswith("Özetleme sırasında beklenmeyen bir hata oluştu:"):
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

def main():
    parser = argparse.ArgumentParser(description="Not Özetleyici")
    parser.add_argument("-t", "--text", help="Özetlenecek metin")
    parser.add_argument("-f", "--file", help="Özetlenecek metin dosyası")
    args = parser.parse_args()

    if not args.text and not args.file:
        print("❌ Lütfen -t ile metin veya -f ile dosya verin.")
        return

    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ Dosya bulunamadı: {args.file}")
            return
        
        file_extension = os.path.splitext(args.file)[1].lower()
        if file_extension == ".txt":
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        elif file_extension == ".pdf":
            text = read_pdf_text(args.file)
            if text.startswith("PDF okuma hatası:"):
                print(f"❌ {text}")
                return
        elif file_extension == ".docx":
            text = read_docx_text(args.file)
            if text.startswith("DOCX okuma hatası:"):
                print(f"❌ {text}")
                return
        else:
            print(f"❌ Desteklenmeyen dosya formatı: {file_extension}. Sadece .txt, .pdf ve .docx desteklenir.")
            return
    else:
        text = args.text

    summary = summarize_long_text(text)
    if not summary.startswith("API isteği sırasında bir hata oluştu:") and        not summary.startswith("API'den beklenmedik yanıt yapısı:") and        not summary.startswith("Yapılandırma hatası:") and        not summary.startswith("Özetleme sırasında beklenmeyen bir hata oluştu:"):
        summary = clean_summary(summary)
    print("\n📌 Özet:\n")
    print(summary)

if __name__ == "__main__":
    main()
