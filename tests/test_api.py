from fastapi.testclient import TestClient
from src.interface.api import app

def test_health_endpoint():
    # Use FastAPI TestClient to test server healthcheck
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
