import spacy
from typing import List, Dict, Any

# Load the English spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model for spaCy...")
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str, lang: str = "en") -> List[Dict[str, Any]]:
    """
    Extracts named entities from the given text using spaCy.
    Currently supports English.
    """
    if lang != "en":
        # For simplicity, only English is supported for now.
        # In a real application, you would load different spaCy models based on language.
        print(f"Warning: Entity extraction not fully supported for language: {lang}. Only English is processed.")
        return []

    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entity_info = {
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "link": None # Placeholder for entity linking
        }
        # Simple entity linking placeholder: if it's a known entity type,
        # you might try to construct a Wikipedia URL or query a knowledge base.
        # This is a very basic example.
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
            # Example: try to link to Wikipedia (very naive)
            # In a real system, this would involve more sophisticated disambiguation
            # and knowledge base lookups.
            search_term = ent.text.replace(" ", "_")
            entity_info["link"] = f"https://en.wikipedia.org/wiki/{search_term}"
        
        entities.append(entity_info)
    return entities
