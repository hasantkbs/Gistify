def read_text_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Metin dosyası bulunamadı: {file_path}") from e
    except Exception as e:
        raise Exception(f"Metin dosyası okuma hatası: {e}") from e
