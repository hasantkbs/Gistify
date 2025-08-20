from docx import Document

def read_docx_text(file_path):
    text = ""
    try:
        document = Document(file_path)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        return f"DOCX okuma hatası: {e}"
    return text

