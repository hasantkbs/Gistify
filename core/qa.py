from transformers import pipeline
from typing import Dict

_QA_PIPELINE = None

def _get_qa_pipeline():
    global _QA_PIPELINE
    if _QA_PIPELINE is None:
        # Using a distilled version for a balance of performance and speed.
        _QA_PIPELINE = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return _QA_PIPELINE

def answer_question_from_summary(question: str, summary: str) -> Dict:
    """
    Answers a question based on a given summary using a QA model.

    :param question: The question to be answered.
    :param summary: The summary to find the answer in.
    :return: A dictionary containing the answer.
    """
    if not question or not summary:
        return {"answer": "Please provide a valid question and summary."}

    qa_pipeline = _get_qa_pipeline()
    result = qa_pipeline(question=question, context=summary)

    return {"answer": result["answer"]}
