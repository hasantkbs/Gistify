from langdetect import detect, DetectorFactory
import re

# Set seed for reproducibility in langdetect
DetectorFactory.seed = 0

# Mapping from langdetect codes to mBART-50 language codes
LANG_CODE_MAP = {
    "en": "en_XX", "tr": "tr_TR", "fr": "fr_XX", "de": "de_DE", "es": "es_XX",
    "it": "it_IT", "ru": "ru_RU", "ar": "ar_AR", "zh-cn": "zh_CN", "pt": "pt_XX",
    "nl": "nl_XX", "ja": "ja_XX", "ko": "ko_KR", "hi": "hi_IN", "ur": "ur_PK",
    "fa": "fa_IR", "bn": "bn_IN", "vi": "vi_VN", "th": "th_TH", "id": "id_ID",
    "ms": "ms_MY", "sw": "sw_KE", "ha": "ha_NG", "pl": "pl_PL", "uk": "uk_UA",
    "ro": "ro_RO", "cs": "cs_CZ", "hu": "hu_HU", "fi": "fi_FI", "sv": "sv_SE",
    "da": "da_DK", "no": "no_NO", "el": "el_GR", "bg": "bg_BG", "sr": "sr_RS",
    "sk": "sk_SK", "sl": "sl_SI", "et": "et_EE", "lv": "lv_LV", "lt": "lt_LT",
    "hr": "hr_HR", "ca": "ca_ES", "eu": "eu_ES", "gl": "gl_ES", "af": "af_ZA",
    "am": "am_ET", "az": "az_AZ", "be": "be_BY", "gu": "gu_IN", "is": "is_IS",
    "ka": "ka_GE", "km": "km_KH", "lo": "lo_LA", "mk": "mk_MK", "ml": "ml_IN",
    "mn": "mn_MN", "my": "my_MM", "ne": "ne_NP", "om": "om_ET", "ps": "ps_AF",
    "so": "so_SO", "sq": "sq_AL", "ta": "ta_IN", "te": "te_IN", "ti": "ti_ET",
    "ug": "ug_CN", "uz": "uz_UZ", "xh": "xh_ZA", "yi": "yi_US", "yo": "yo_NG",
    "zu": "zu_ZA",
}

# Constants for chunking
MAX_CHUNK_CHARS = 3000 # Approximate character limit for a chunk
MIN_OVERLAP_CHARS = 200 # Overlap between chunks to maintain context

def clean_summary(summary_text):
    # Remove common English adverbs that might be hallucinated
    summary_text = re.sub(r'\b(guiltful|grotesque|enthusiastically|worrisome|tahminiably)\b', '', summary_text, flags=re.IGNORECASE)
    # Remove non-Turkish characters (keep Turkish alphabet, numbers, punctuation, and spaces)
    summary_text = re.sub(r"[^a-zA-Z0-9ğĞüÜşŞıİöÖçÇ\s.,;!?\'\"()[\\]-]", '', summary_text)
    # Clean up multiple spaces
    summary_text = re.sub(r'\s+', ' ', summary_text).strip()
    return summary_text

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
