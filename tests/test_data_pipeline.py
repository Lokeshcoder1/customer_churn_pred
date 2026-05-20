"""
Tests for Data Pipeline
Tests data loading, cleaning, and preprocessing.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.data_pipeline import DataPipeline
from src.config import TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    df = pd.DataFrame({
        "customerID": ["001", "002", "003", "004", "005"],
        "tenure": [24, 48, 12, 36, 6],
        "MonthlyCharges": [65.5, 45.0, 75.5, 55.0, 85.0],
        "TotalCharges": [1570.0, 2160.0, 906.0, 1980.0, 510.0],
        "gender": ["M", "F", "M", "F", "M"],
        "SeniorCitizen": [0, 1, 0, 0, 1],
        "Partner": ["Yes", "No", "Yes", "Yes", "No"],
        "Churn": ["No", "Yes", "No", "Yes", "No"]
    })
    return df


@pytest.fixture
def temp_data_file(sample_data):
    """Create temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_data.to_csv(f, index=False)
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink()


def test_load_raw_success(temp_data_file):
    """Test loading raw data successfully."""
    pipeline = DataPipeline()
    df = pipeline.load_raw(temp_data_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert list(df.columns) == ["customerID", "tenure", "MonthlyCharges", "TotalCharges",
                                "gender", "SeniorCitizen", "Partner", "Churn"]


def test_load_raw_file_not_found():
    """Test error handling when file doesn't exist."""
    pipeline = DataPipeline()

    with pytest.raises(FileNotFoundError):
        pipeline.load_raw("nonexistent_file.csv")


def test_clean_handles_missing_values():
    """Test that clean handles missing values."""
    pipeline = DataPipeline()

    df = pd.DataFrame({
        "tenure": [10, 20, np.nan, 30],
        "MonthlyCharges": [50.0, 60.0, 70.0, np.nan],
        "TotalCharges": [500.0, 1200.0, 1400.0, 2000.0],
        "Churn": [0, 1, 0, 1]
    })

    df_clean = pipeline.clean(df)

    # Should fill missing TotalCharges
    assert df_clean.isnull().any().any() == 0


def test_clean_standardizes_churn():
    """Test that churn values are converted to binary."""
    pipeline = DataPipeline()

    df = pd.DataFrame({
        "tenure": [10, 20, 30],
        "MonthlyCharges": [50.0, 60.0, 70.0],
        "TotalCharges": [500.0, 1200.0, 2100.0],
        "Churn": ["Yes", "No", "Yes"]
    })

    df_clean = pipeline.clean(df)

    # Churn should be 0 or 1
    assert set(df_clean["Churn"].unique()) == {0, 1}
    assert df_clean["Churn"].dtype in [np.int64, np.int32, int]


def test_encode_categorical_fit():
    """Test one-hot encoding (fit mode)."""
    pipeline = DataPipeline()

    df = pd.DataFrame({
        "tenure": [10, 20, 30],
        "MonthlyCharges": [50.0, 60.0, 70.0],
        "TotalCharges": [500.0, 1200.0, 2100.0],
        "gender": ["M", "F", "M"],
        "Partner": ["Yes", "No", "Yes"],
        "Churn": [0, 1, 0]
    })

    df_encoded = pipeline.encode_categorical(df, fit=True)

    # Should have dropped original categorical columns
    assert "gender" not in df_encoded.columns
    assert "Partner" not in df_encoded.columns

    # Should have new binary columns
    encoded_columns = [col for col in df_encoded.columns if col.startswith("gender_") or col.startswith("Partner_")]
    assert len(encoded_columns) > 0


def test_scale_numeric_fit():
    """Test numeric scaling (fit mode)."""
    pipeline = DataPipeline()

    df = pd.DataFrame({
        "tenure": [10, 20, 30],
        "MonthlyCharges": [50.0, 60.0, 70.0],
        "TotalCharges": [500.0, 1200.0, 2100.0],
        "Churn": [0, 1, 0]
    })

    df_scaled = pipeline.scale_numeric(df, fit=True)

    # After scaling, mean should be ~0, std ~1
    for col in NUMERIC_FEATURES:
        if col in df_scaled.columns:
            assert abs(df_scaled[col].mean()) < 0.1, f"{col} mean not ~0"
            assert abs(df_scaled[col].std(ddof=0) - 1.0) < 0.1, f"{col} std not ~1"


def test_split_data_stratification():
    """Test that train/test split is stratified."""
    pipeline = DataPipeline()

    # Create imbalanced data
    df = pd.DataFrame({
        "tenure": list(range(100)),
        "MonthlyCharges": np.random.uniform(30, 100, 100),
        "TotalCharges": np.random.uniform(300, 3000, 100),
        "gender": np.random.choice(["M", "F"], 100),
        "Churn": [1] * 25 + [0] * 75  # 25% churn
    })

    df_encoded = pd.get_dummies(df, columns=["gender"], drop_first=True)

    X_train, X_test, y_train, y_test = pipeline.split_data(df_encoded)

    # Check stratification (class distribution similar)
    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()

    # Should be close to 25% churn in both
    assert abs(train_churn_rate - 0.25) < 0.05
    assert abs(test_churn_rate - 0.25) < 0.05


def test_pipeline_save_artifacts():
    """Test saving pipeline artifacts (scaler, encoder)."""
    pipeline = DataPipeline()

    df = pd.DataFrame({
        "tenure": [10, 20, 30],
        "MonthlyCharges": [50.0, 60.0, 70.0],
        "TotalCharges": [500.0, 1200.0, 2100.0],
        "gender": ["M", "F", "M"],
        "Churn": [0, 1, 0]
    })

    # Fit preprocessing
    pipeline.encode_categorical(df, fit=True)
    pipeline.scale_numeric(df, fit=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "artifacts.pkl"
        pipeline.save_pipeline(save_path)

        # Check file exists
        assert save_path.exists()

        # Load and verify
        pipeline2 = DataPipeline()
        pipeline2.load_pipeline(save_path)

        assert pipeline2.scaler is not None
        assert pipeline2.categorical_encoder is not None