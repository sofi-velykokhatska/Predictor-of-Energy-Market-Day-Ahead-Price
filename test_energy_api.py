import os
import pytest

# The API_KEY must be set in the environment BEFORE importing energy_api,
# because energy_api.py reads it at import time (API_KEY = os.environ.get("API_KEY")).
# If we imported first and set the env var after, the app would still think
# no key is configured.
os.environ["API_KEY"] = "test-key-123"

from energy_api import app

VALID_PAYLOAD = {
    "wind_power": 5000,
    "solar_proxy": 800,
    "heating_degree": 5,
    "cooling_degree": 0,
    "precipitation": 0.5,
    "hour": 14,
    "month": 6,
    "is_weekend": 0,
    "price_lag_24": 150,
    "price_lag_168": 145,
    "gas_price": 45,
}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check_returns_200(client):
    """/health must stay unauthenticated and report status for platform health checks."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "healthy"


def test_predict_without_api_key_is_rejected(client):
    """No X-API-Key header at all -> should be blocked before the model ever runs."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 401


def test_predict_with_wrong_api_key_is_rejected(client):
    """An incorrect key must be treated the same as no key -> still blocked."""
    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_predict_with_correct_api_key_succeeds(client):
    """The core happy path: valid key + valid input -> a real prediction back."""
    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
        headers={"X-API-Key": "test-key-123"},
    )
    print(response.get_json())
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert isinstance(data["predicted_price_eur_mwh"], float)


def test_predict_missing_features_returns_400(client):
    """Even with a valid key, incomplete input should be rejected with a clear error,
    not silently passed to the model."""
    incomplete_payload = {"wind_power": 5000}
    response = client.post(
        "/predict",
        json=incomplete_payload,
        headers={"X-API-Key": "test-key-123"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid request data"


def test_features_endpoint_requires_api_key(client):
    """/features exposes model metadata and must be protected like /predict."""
    response = client.get("/features")
    assert response.status_code == 401


def test_features_endpoint_with_api_key(client):
    """With a valid key, /features should return the expected feature list."""
    response = client.get("/features", headers={"X-API-Key": "test-key-123"})
    assert response.status_code == 200
    assert "features" in response.get_json()