import pypdf

def read_pdf_text(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text() or ""
    return text
