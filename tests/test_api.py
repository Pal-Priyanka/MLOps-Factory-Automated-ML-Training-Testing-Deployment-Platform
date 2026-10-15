"""Integration tests for FastAPI inference endpoints, input validation, and operational metrics."""

import pytest
from fastapi.testclient import TestClient

from mlops_factory.serving.app import create_app, model_container


@pytest.fixture
def client_with_mock_model(fitted_mock_pipeline):
    """Sets up FastAPI test client with an initialized mock model."""
    app = create_app()

    with TestClient(app) as test_client:
        # Set mock container state after lifespan completes
        model_container.pipeline = fitted_mock_pipeline
        model_container.metadata = {
            "model_name": "Mock_HistGradientBoosting",
            "trained_at_utc": "2026-10-15T00:00:00Z",
            "model_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "features": {"total_features": 36},
            "environment": {"scikit_learn_version": "1.9.1"},
            "quality_gates": {"passed": True, "checks": []},
        }
        model_container.is_ready = True
        yield test_client


def test_health_endpoint(client_with_mock_model):
    """Validates /health endpoint status and uptime."""
    response = client_with_mock_model.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data
    assert "X-Request-ID" in response.headers
    assert "X-Response-Time-Ms" in response.headers


def test_ready_endpoint(client_with_mock_model):
    """Validates /ready endpoint when model is loaded."""
    response = client_with_mock_model.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_ready_endpoint_unready():
    """Validates /ready endpoint returns 503 when model is not ready."""
    app = create_app()

    with TestClient(app) as client:
        # Override container to unready state after startup
        model_container.is_ready = False
        model_container.init_error = "Model artifact missing"
        response = client.get("/ready")
        assert response.status_code == 503
        assert "Service unready" in response.json()["detail"]


def test_metadata_endpoint(client_with_mock_model):
    """Validates /metadata returns model governance and version info."""
    response = client_with_mock_model.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "Mock_HistGradientBoosting"
    assert data["total_features"] == 36
    assert data["quality_gate_passed"] is True


def test_predict_endpoint_valid_payload(client_with_mock_model, sample_valid_customer_payload):
    """Validates single prediction endpoint on realistic input."""
    payload = {"customer": sample_valid_customer_payload}
    response = client_with_mock_model.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert isinstance(data["churn_prediction"], bool)
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_tier"] in ["Low", "Medium", "High"]
    assert "recommended_action" in data
    assert data["latency_ms"] >= 0.0


def test_predict_endpoint_invalid_payload(client_with_mock_model, sample_valid_customer_payload):
    """Validates 422 Unprocessable Entity error handling on malformed input."""
    # Corrupt age with invalid negative number
    bad_payload = dict(sample_valid_customer_payload)
    bad_payload["Age"] = -25

    response = client_with_mock_model.post("/predict", json={"customer": bad_payload})
    assert response.status_code == 422
    err_data = response.json()
    assert err_data["error"] == "Validation Error"
    assert "details" in err_data


def test_batch_predict_endpoint(client_with_mock_model, sample_valid_customer_payload):
    """Validates batch prediction endpoint on collection of customers."""
    batch_customers = [sample_valid_customer_payload, sample_valid_customer_payload]
    response = client_with_mock_model.post("/predict/batch", json={"customers": batch_customers})

    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2
    assert "batch_latency_ms" in data


def test_metrics_json_endpoint(client_with_mock_model, sample_valid_customer_payload):
    """Validates /metrics endpoint returns latency percentiles and request counts."""
    # Issue a prediction first to populate telemetry
    client_with_mock_model.post("/predict", json={"customer": sample_valid_customer_payload})

    response = client_with_mock_model.get("/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert metrics["total_requests"] >= 1
    assert "latency_ms" in metrics
    assert "p50_ms" in metrics["latency_ms"]
    assert "p95_ms" in metrics["latency_ms"]


def test_metrics_prometheus_endpoint(client_with_mock_model):
    """Validates /metrics/prometheus returns valid Prometheus plaintext exposition format."""
    response = client_with_mock_model.get("/metrics/prometheus")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    text = response.text
    assert "mlops_inference_requests_total" in text
    assert "mlops_service_uptime_seconds" in text
