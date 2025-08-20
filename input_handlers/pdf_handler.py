import PyPDF2

def read_pdf_text(file_path):
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() + "\n"
    except Exception as e:
        return f"PDF okuma hatası: {e}"
    return text
