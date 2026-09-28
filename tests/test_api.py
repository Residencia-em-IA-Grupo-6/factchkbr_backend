from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "active_analyzers" in data
    assert "available_analyzers" in data


def test_analyze_contract():
    payload = {
        "text": "Texto de notícia a ser verificado pelos modelos.",
        "urls": ["https://exemplo.com/noticia"],
        "user_id": 123456789,
        "chat_id": 987654321
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "claim" in data
    assert "verdict" in data
    assert "confidence" in data
    assert "summary" in data
    assert "reasons" in data
    assert "sources" in data
