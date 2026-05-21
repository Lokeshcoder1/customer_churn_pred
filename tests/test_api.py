"""
Tests for FastAPI API
Tests endpoints, validation, and error handling.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Create a mock model for testing
def create_mock_model():
    """Create a realistic mock pipeline that handles both encoding and scaling."""
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    import pandas as pd

    # Create sample training data (realistic features)
    X_mock = pd.DataFrame({
        # Numeric
        'tenure': [10, 20, 30],
        'monthly_charges': [50.0, 60.0, 70.0],
        'total_charges': [500.0, 1200.0, 2100.0],
        # Categorical (will be one-hot encoded)
        'contract': ['Month-to-month', 'One year', 'Two year'],
        'internet_service': ['DSL', 'Fiber optic', 'DSL'],
        'gender': ['Male', 'Female', 'Male'],
        'partner': ['Yes', 'No', 'Yes'],
    })

    y_mock = [0, 1, 0]

    # Create preprocessing pipeline
    numeric_features = ['tenure', 'monthly_charges', 'total_charges']
    categorical_features = ['contract', 'internet_service', 'gender', 'partner']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

    # Create full pipeline with preprocessing + model
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(random_state=42, max_iter=1000))
    ])

    pipeline.fit(X_mock, y_mock)

    return pipeline


@pytest.fixture
def mock_model_path():
    """Create a temporary model file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        model = create_mock_model()
        joblib.dump(model, f.name)
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def client(mock_model_path, monkeypatch):
    """Create test client with mocked model path."""
    # Patch the model path
    monkeypatch.setenv("MODEL_PATH", mock_model_path)

    from src.api import app, ml_pipeline

    # Load the mock model
    ml_pipeline.load(mock_model_path)

    return TestClient(app)


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