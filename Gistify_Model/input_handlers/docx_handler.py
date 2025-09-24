from docx import Document

def read_docx_text(file_path: str) -> str:
    text = ""
    try:
        document = Document(file_path)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        raise Exception(f"DOCX okuma hatası: {e}") from e
    return text

