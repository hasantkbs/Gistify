import PyPDF2

def read_pdf_text(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    except FileNotFoundError as e:
        raise FileNotFoundError(f"PDF dosyası bulunamadı: {file_path}") from e
    except Exception as e:
        raise Exception(f"PDF okuma hatası: {e}") from e
