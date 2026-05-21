"""
Combined ML Pipeline
Integrates data preprocessing and model into a single production object.
"""

import joblib
import logging
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import MODELS_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class MLPipeline:
    """
    Complete ML pipeline: preprocessing + model in a single object.

    When you load this pipeline, it automatically:
    - Applies scaling to numeric features
    - Applies encoding to categorical features
    - Makes predictions with the trained model

    This is what gets serialized and deployed.

    Attributes:
        pipeline: sklearn Pipeline combining preprocessing and model
    """

    def __init__(self):
        """Initialize with empty pipeline."""
        self.pipeline = None
        logger.info("MLPipeline initialized")

    def set_pipeline(self, pipeline: Pipeline) -> None:
        """
        Set the sklearn pipeline.

        Args:
            pipeline: Fitted sklearn Pipeline object
        """
        self.pipeline = pipeline
        logger.info("Pipeline set successfully")

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make binary predictions.

        Args:
            X: Input features (DataFrame or array)

        Returns:
            Binary predictions (0 or 1)

        Raises:
            ValueError: If pipeline not fitted
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call set_pipeline() first.")

        try:
            return self.pipeline.predict(X)
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            raise

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make probability predictions.

        Args:
            X: Input features

        Returns:
            Array of shape (n_samples, 2) with probabilities for [no_churn, churn]

        Raises:
            ValueError: If pipeline not fitted
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call set_pipeline() first.")

        try:
            return self.pipeline.predict_proba(X)
        except Exception as e:
            logger.error(f"Error in predict_proba: {str(e)}")
            raise

    def predict_with_confidence(self, X: Union[pd.DataFrame, np.ndarray],
                                threshold: float = 0.5) -> dict:
        """
        Make predictions with confidence levels.

        Args:
            X: Input features
            threshold: Decision threshold (default 0.5)

        Returns:
            Dict with:
                - predictions: Binary predictions
                - probabilities: Churn probabilities (0-1)
                - confidence: Confidence level ("high", "medium", "low")
        """
        proba = self.predict_proba(X)
        churn_proba = proba[:, 1]
        predictions = (churn_proba > threshold).astype(int)

        # Confidence based on distance from threshold
        confidence_scores = np.abs(churn_proba - threshold)
        confidence_levels = np.where(
            confidence_scores > 0.2, "high",
            np.where(confidence_scores > 0.1, "medium", "low")
        )

        return {
            "predictions": predictions,
            "probabilities": churn_proba,
            "confidence": confidence_levels
        }

    def save(self, filepath: Optional[Path] = None) -> None:
        """
        Serialize pipeline to disk.

        Args:
            filepath: Where to save. Default: models/production_model.pkl
        """
        if filepath is None:
            filepath = MODELS_DIR / "production_model.pkl"

        if self.pipeline is None:
            raise ValueError("No pipeline to save. Train a model first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pipeline, filepath)
        logger.info(f"Saved pipeline to {filepath}")
        print(f"✓ Model saved: {filepath}")

    def load(self, filepath: Optional[Path] = None) -> None:
        """
        Load pipeline from disk.

        Args:
            filepath: Where to load from. Default: models/production_model.pkl

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if filepath is None:
            filepath = MODELS_DIR / "production_model.pkl"

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model not found: {filepath}")

        self.pipeline = joblib.load(filepath)
        logger.info(f"Loaded pipeline from {filepath}")
        print(f"✓ Model loaded: {filepath}")

    def model_summary(self) -> str:
        """
        Return a text summary of the pipeline.

        Returns:
            String describing the pipeline structure
        """
        if self.pipeline is None:
            return "Pipeline not fitted"

        summary = "\n" + "=" * 70 + "\n"
        summary += "ML PIPELINE SUMMARY\n"
        summary += "=" * 70 + "\n\n"

        summary += "Pipeline Steps:\n"
        summary += "-" * 70 + "\n"
        for i, (name, step) in enumerate(self.pipeline.steps, 1):
            summary += f"{i}. {name}\n"
            summary += f"   Type: {type(step).__name__}\n"

            # Add details for specific steps
            if hasattr(step, "get_params"):
                params = step.get_params()
                if params:
                    summary += f"   Parameters: {len(params)} configured\n"

        summary += "\n" + "=" * 70 + "\n"

        return summary