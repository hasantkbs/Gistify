from transformers import pipeline
from typing import Dict

_SENTIMENT_PIPELINE = None

def _get_sentiment_pipeline():
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is None:
        _SENTIMENT_PIPELINE = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return _SENTIMENT_PIPELINE

def analyze_sentiment(text: str) -> Dict:
    """
    Analyzes the sentiment of a given text.

    :param text: The text to be analyzed.
    :return: A dictionary containing the sentiment label and score.
    """
    if not text:
        return {"label": "neutral", "score": 0.0}

    sentiment_pipeline = _get_sentiment_pipeline()
    result = sentiment_pipeline(text)

    # The result is a list of dictionaries, we take the first one
    return result[0]
