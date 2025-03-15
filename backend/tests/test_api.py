import requests

BASE_URL = "http://127.0.0.1:5000"  # Ensure Flask is running

def test_lime_explanation():
    """Ensure LIME Explanation API works."""
    response = requests.post(
        f"{BASE_URL}/api/lime_explanation",
        json={"text": "Machine learning is the future."}
    )
    assert response.status_code == 200
    assert "<html>" in response.text  # Ensure it returns an HTML page

def test_api_reachability():
    """Check if the API is online."""
    response = requests.post(f"{BASE_URL}/api/classify", json={"text": "Test text"})
    assert response.status_code == 200

def test_classify_missing_text():
    """Send empty request and expect 400 error."""
    response = requests.post(f"{BASE_URL}/api/classify", json={})
    assert response.status_code == 400

def test_classify_valid_text():
    """Test classification endpoint with valid input."""
    response = requests.post(
        f"{BASE_URL}/api/classify",
        json={"text": "This is an AI-generated article about deep learning."}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "explanation" in data
    assert data["prediction"] in ["Human-written", "AI-generated"]   
    
     