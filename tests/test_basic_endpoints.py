import requests


BASE_URL = "http://localhost:8000"


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"


def test_retrieve_requires_query():
    r = requests.get(f"{BASE_URL}/retrieve")
    assert r.status_code == 422 or r.status_code == 400


def test_stats():
    r = requests.get(f"{BASE_URL}/stats")
    assert r.status_code in (200, 500)
    # If 200, expect json with some keys
    if r.status_code == 200:
        data = r.json()
        assert "vector_store_path" in data
