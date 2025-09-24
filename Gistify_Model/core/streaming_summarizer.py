from core.summarizer import summarize_text

class StreamingSummarizer:
    def __init__(self, length="medium", summary_type="abstractive"):
        self.text = ""
        self.length = length
        self.summary_type = summary_type

    def add_chunk(self, chunk: str):
        self.text += chunk

    def get_summary(self):
        if not self.text:
            return ""
        
        # For now, we re-summarize the whole text on each call.
        # This can be optimized later.
        result = summarize_text(self.text, length=self.length, summary_type=self.summary_type)
        return result.get("summary", "")
