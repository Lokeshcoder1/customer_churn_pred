"""
Churn Prediction Package
Production-grade ML system for predicting customer churn.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Customer Churn Prediction API"

from .config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, ARTIFACTS_DIR,
    TEST_SIZE, RANDOM_STATE, TARGET_COLUMN
)
from .data_pipeline import DataPipeline
from .model_comparison import ModelComparison
from .pipeline import MLPipeline

__all__ = [
    "DataPipeline",
    "ModelComparison",
    "MLPipeline",
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "ARTIFACTS_DIR"
]