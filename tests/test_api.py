import pytest
from fastapi.testclient import TestClient
from api.main import app  # Import the FastAPI app instance
import httpx # Import httpx for making requests

client = TestClient(app)

# Fixture to create a test user and get an access token
@pytest.fixture(scope="module")
def test_user_token():
    # Register a test user
    user_data = {"email": "test@example.com", "password": "testpassword"}
    client.post("/auth/register", json=user_data)

    # Log in the test user to get a token
    token_response = client.post(
        "/auth/login",
        data={"username": user_data["email"], "password": user_data["password"]}
    )
    assert token_response.status_code == 200
    token = token_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_health_check():
    """
    Tests the /health endpoint.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_summarize_text_success(test_user_token):
    """
    Tests the /summarize endpoint with valid text.
    """
    # This is a basic test. For a real-world scenario, 
    # you might mock the summarizer to avoid long processing times.
    response = client.post(
        "/summarize",
        json={"text": "This is a test sentence for summarization."},
        headers=test_user_token
    )
    assert response.status_code == 202
    assert "task_id" in response.json()
    assert response.json()["status"] == "queued"

def test_summarize_text_no_text(test_user_token):
    """
    Tests the /summarize endpoint when no text is provided.
    """
    response = client.post(
        "/summarize",
        json={"text": ""},
        headers=test_user_token
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "No text provided for summarization.", "error_type": "EmptyContentError"}

def test_summarize_file_unsupported_format(test_user_token):
    """
    Tests the /summarize_file endpoint with an unsupported file type.
    """
    # Create a dummy file with an unsupported extension
    with open("test.unsupported", "w") as f:
        f.write("dummy content")
    
    with open("test.unsupported", "rb") as f:
        response = client.post(
            "/summarize_file",
            files={"file": ("test.unsupported", f, "text/plain")},
            headers=test_user_token
        )
    
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_get_usage_stats(test_user_token):
    """
    Tests the /usage endpoint to ensure it returns correct usage statistics.
    """
    response = client.get("/usage", headers=test_user_token)
    assert response.status_code == 200
    data = response.json()
    assert "total_gists" in data
    assert "gists_last_30_days" in data
    assert "daily_gists_last_7_days" in data
    assert isinstance(data["total_gists"], int)
    assert isinstance(data["gists_last_30_days"], int)
    assert isinstance(data["daily_gists_last_7_days"], dict)
