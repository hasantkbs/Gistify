import pytest
from fastapi.testclient import TestClient
from api.main import app  # Import the FastAPI app instance

client = TestClient(app)

def test_health_check():
    """
    Tests the /health endpoint.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_summarize_text_success():
    """
    Tests the /summarize endpoint with valid text.
    """
    # This is a basic test. For a real-world scenario, 
    # you might mock the summarizer to avoid long processing times.
    response = client.post("/summarize", json={"text": "This is a test sentence for summarization."})
    assert response.status_code == 202
    assert "task_id" in response.json()
    assert response.json()["status"] == "Processing"

def test_summarize_text_no_text():
    """
    Tests the /summarize endpoint when no text is provided.
    """
    response = client.post("/summarize", json={"text": ""})
    assert response.status_code == 400
    assert response.json() == {"detail": "No text provided for summarization."}

def test_summarize_file_unsupported_format():
    """
    Tests the /summarize_file endpoint with an unsupported file type.
    """
    # Create a dummy file with an unsupported extension
    with open("test.unsupported", "w") as f:
        f.write("dummy content")
    
    with open("test.unsupported", "rb") as f:
        response = client.post("/summarize_file", files={"file": ("test.unsupported", f, "text/plain")})
    
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]
