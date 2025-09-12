import pytest
from core.summarizer import _preclean_text, split_text_into_chunks, _pick_model_for_lang
from config.settings import ENGLISH_MODEL_NAME, TURKISH_MODEL_NAME, MT5_MODEL_NAME

def test_preclean_text():
    """
    Tests the _preclean_text function.
    """
    assert _preclean_text("  hello   world  ") == "hello world"
    assert _preclean_text("hello-\nworld") == "helloworld"
    assert _preclean_text("hello\nworld") == "hello world"
    assert _preclean_text("hello  ,  world") == "hello, world"
    assert _preclean_text("word word word word") == "word" # Test deduplication
    assert _preclean_text(None) == ""

def test_split_text_into_chunks():
    """
    Tests the split_text_into_chunks function.
    """
    text = "This is the first sentence. This is the second sentence. This is the third sentence."
    chunks = split_text_into_chunks(text, max_chars=60)
    assert len(chunks) == 2
    assert chunks[0] == "This is the first sentence. This is the second sentence."
    assert chunks[1] == "This is the third sentence."

    # Test with a single long sentence
    long_sentence = "This is a very long sentence that should not be split."
    chunks_long = split_text_into_chunks(long_sentence, max_chars=20)
    assert len(chunks_long) == 1
    assert chunks_long[0] == long_sentence

def test_pick_model_for_lang():
    """
    Tests the _pick_model_for_lang function.
    """
    assert _pick_model_for_lang("tr") == TURKISH_MODEL_NAME
    assert _pick_model_for_lang("en") == ENGLISH_MODEL_NAME
    assert _pick_model_for_lang("de") == MT5_MODEL_NAME
    assert _pick_model_for_lang("fr") == MT5_MODEL_NAME
    assert _pick_model_for_lang(None) == MT5_MODEL_NAME
