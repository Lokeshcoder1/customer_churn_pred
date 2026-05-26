"""
Tests for FastAPI API
Tests endpoints, validation, and error handling.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import joblib
from sklearn.linear_model import LogisticRegression

from src.data_pipeline import DataPipeline


# Create a mock model for testing
def create_mock_model(temp_dir):
    """Create a mock production-style model and preprocessing artifacts."""
    import pandas as pd

    raw_df = pd.DataFrame({
        "customerID": ["001", "002", "003", "004", "005", "006"],
        "tenure": [10, 20, 30, 40, 50, 60],
        "MonthlyCharges": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "TotalCharges": [500.0, 1200.0, 2100.0, 3200.0, 4500.0, 6000.0],
        "gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 0, 1, 0],
        "Partner": ["Yes", "No", "Yes", "No", "Yes", "No"],
        "Dependents": ["No", "Yes", "No", "Yes", "No", "Yes"],
        "PhoneService": ["Yes", "Yes", "No", "Yes", "No", "Yes"],
        "MultipleLines": ["No", "Yes", "No phone service", "Yes", "No", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "No internet service", "Yes", "No", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No internet service", "No", "Yes", "No internet service"],
        "DeviceProtection": ["Yes", "No", "No internet service", "Yes", "No", "No internet service"],
        "TechSupport": ["No", "Yes", "No internet service", "No", "Yes", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service", "Yes", "No", "No internet service"],
        "StreamingMovies": ["Yes", "No", "No internet service", "Yes", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Credit card", "Electronic check", "Mailed check"],
        "Churn": ["No", "Yes", "No", "Yes", "No", "Yes"]
    })

    raw_path = Path(temp_dir) / "raw_data.csv"
    raw_df.to_csv(raw_path, index=False)

    pipeline = DataPipeline()
    X_train, X_test, y_train, y_test = pipeline.process(raw_path, save=False)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    artifacts_path = Path(temp_dir) / "pipeline_artifacts.pkl"
    pipeline.save_pipeline(artifacts_path)

    return model, artifacts_path


@pytest.fixture
def mock_model_path():
    """Create temporary model and pipeline artifacts for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model, artifacts_path = create_mock_model(tmpdir)
        model_path = Path(tmpdir) / "production_model.pkl"
        joblib.dump(model, model_path)
        yield str(model_path), artifacts_path


@pytest.fixture
def client(mock_model_path):
    """Create test client with mocked model path and preprocessing artifacts."""
    model_path, artifacts_path = mock_model_path

    from src import api

    api.MODEL_ARTIFACT_PATH = Path(model_path)

    pipeline = DataPipeline()
    pipeline.load_pipeline(artifacts_path)
    pipeline.load_pipeline = lambda filepath=None: None
    api.data_pipeline = pipeline

    api.ml_pipeline.load(model_path)

    return TestClient(api.app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_success(self, client):
        """Test successful health check."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] in ["healthy", "unhealthy"]
        assert "model_loaded" in response.json()


class TestPredictionEndpoint:
    """Tests for single prediction endpoint."""

    def test_predict_success(self, client):
        """Test successful prediction."""
        payload = {
            "tenure": 24,
            "monthly_charges": 65.5,
            "total_charges": 1570.0,
            "contract": "Two year",
            "internet_service": "Fiber optic",
            "online_security": "Yes",
            "online_backup": "No",
            "device_protection": "Yes",
            "tech_support": "Yes",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
            "gender": "Female",
            "phone_service": "Yes",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check"
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "confidence" in data

        # Validate ranges
        assert 0 <= data["churn_probability"] <= 1
        assert isinstance(data["churn_prediction"], bool)
        assert data["confidence"] in ["high", "medium", "low"]

    def test_predict_invalid_tenure(self, client):
        """Test validation error for invalid tenure."""
        payload = {
            "tenure": -5,  # Invalid: must be > 0
            "monthly_charges": 65.5,
            "total_charges": 1570.0,
            "contract": "Two year",
            "internet_service": "Fiber optic",
            "online_security": "Yes",
            "online_backup": "No",
            "device_protection": "Yes",
            "tech_support": "Yes",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
            "gender": "Female",
            "phone_service": "Yes",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check"
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 422  # Validation error


class TestBatchPredictionEndpoint:
    """Tests for batch prediction endpoint."""

    def test_batch_predict_success(self, client):
        """Test successful batch prediction."""
        payload = {
            "customers": [
                {
                    "tenure": 24,
                    "monthly_charges": 65.5,
                    "total_charges": 1570.0,
                    "contract": "Two year",
                    "internet_service": "Fiber optic",
                    "online_security": "Yes",
                    "online_backup": "No",
                    "device_protection": "Yes",
                    "tech_support": "Yes",
                    "senior_citizen": 0,
                    "partner": "Yes",
                    "dependents": "No",
                    "gender": "Female",
                    "phone_service": "Yes",
                    "streaming_tv": "No",
                    "streaming_movies": "No",
                    "paperless_billing": "Yes",
                    "payment_method": "Electronic check"
                },
                {
                    "tenure": 12,
                    "monthly_charges": 45.0,
                    "total_charges": 540.0,
                    "contract": "Month-to-month",
                    "internet_service": "DSL",
                    "online_security": "No",
                    "online_backup": "Yes",
                    "device_protection": "No",
                    "tech_support": "No",
                    "senior_citizen": 1,
                    "partner": "No",
                    "dependents": "Yes",
                    "gender": "Male",
                    "phone_service": "No",
                    "streaming_tv": "Yes",
                    "streaming_movies": "Yes",
                    "paperless_billing": "No",
                    "payment_method": "Bank transfer"
                }
            ]
        }

        response = client.post("/predict-batch", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert "predictions" in data
        assert "count" in data
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

        # Validate each prediction
        for pred in data["predictions"]:
            assert "churn_probability" in pred
            assert "churn_prediction" in pred
            assert "confidence" in pred
            assert 0 <= pred["churn_probability"] <= 1

    def test_batch_predict_empty(self, client):
        """Test error on empty batch."""
        payload = {"customers": []}

        response = client.post("/predict-batch", json=payload)

        assert response.status_code == 400  # Bad request


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""

    def test_model_info_success(self, client):
        """Test successful model info retrieval."""
        response = client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns welcome message."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "redoc" in data