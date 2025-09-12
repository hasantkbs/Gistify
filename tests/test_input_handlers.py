import pytest
from unittest.mock import patch, MagicMock

from input_handlers.text_handler import read_text_file
from input_handlers.docx_handler import read_docx_text
from input_handlers.pdf_handler import read_pdf_text

# --- Test for text_handler ---

def test_read_text_file():
    """
    Tests reading a .txt file.
    """
    file_path = "tests/test_files/dummy.txt"
    content = read_text_file(file_path)
    assert "This is a test file." in content
    assert "It has multiple lines." in content

def test_read_text_file_not_found():
    """
    Tests reading a non-existent .txt file.
    """
    with pytest.raises(FileNotFoundError): # Changed to FileNotFoundError
        read_text_file("non_existent_file.txt")

# --- Mocks for PDF and DOCX handlers ---

# Mock for python-docx
class MockParagraph:
    def __init__(self, text):
        self.text = text

class MockDocument:
    def __init__(self, paragraphs):
        self.paragraphs = [MockParagraph(p) for p in paragraphs]

# Mock for PyPDF2
class MockPdfReader:
    def __init__(self, pages):
        self.pages = [MagicMock(extract_text=lambda p=p: p) for p in pages]

# --- Tests for docx_handler ---

@patch('input_handlers.docx_handler.Document') # Corrected patch path
def test_read_docx_text(mock_document):
    """
    Tests reading a .docx file by mocking the docx library.
    """
    mock_document.return_value = MockDocument(["This is a docx test.", "It has paragraphs."])
    
    content = read_docx_text("dummy.docx")
    assert content == "This is a docx test.\nIt has paragraphs.\n"

def test_read_docx_text_not_found():
    """
    Tests reading a non-existent .docx file.
    """
    with pytest.raises(Exception): # docx library raises a generic Exception for file not found
        read_docx_text("non_existent_file.docx")

# --- Tests for pdf_handler ---

@patch('builtins.open', new_callable=MagicMock)
@patch('input_handlers.pdf_handler.PyPDF2.PdfReader')
def test_read_pdf_text(mock_pdf_reader, mock_open):
    """
    Tests reading a .pdf file by mocking the PyPDF2 library and file open.
    """
    mock_pdf_reader.return_value = MockPdfReader(["This is a pdf test.", "It has pages."])
    
    content = read_pdf_text("dummy.pdf")
    assert content == "This is a pdf test.\nIt has pages.\n"


def test_read_pdf_text_not_found():
    """
    Tests reading a non-existent .pdf file.
    """
    with pytest.raises(FileNotFoundError): # pdf_handler now raises FileNotFoundError
        read_pdf_text("non_existent_file.pdf")