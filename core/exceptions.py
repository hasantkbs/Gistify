"""
Custom exceptions for the Gistify application.
"""

class GistifyError(Exception):
    """Base exception class for the application."""
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(self.detail)

class UnsupportedFileTypeError(GistifyError):
    """Raised when a file format is not supported."""
    pass

class FileProcessingError(GistifyError):
    """Raised when there is an error processing a file (e.g., reading, decoding)."""
    pass

class EmptyContentError(GistifyError):
    """Raised when the input text or file is empty."""
    pass

class UrlConnectionError(GistifyError):
    """Raised when a URL cannot be reached or read."""
    pass

class SummarizationError(GistifyError):
    """Raised for errors during the summarization process."""
    pass
