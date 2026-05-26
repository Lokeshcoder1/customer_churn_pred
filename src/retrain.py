"""
Automated Retraining Pipeline
Monitors data drift, retrains model, and manages versions.

Run via cron: 0 0 * * 0 (weekly on Sunday)
Or via Airflow, APScheduler, GitHub Actions
"""

import logging
import json
import pickle
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import requests
from sklearn.metrics import roc_auc_score

from src.config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR,
    CV_FOLDS, RANDOM_STATE, TARGET_COLUMN
)
from src.data_pipeline import DataPipeline
from src.model_comparison import ModelComparison

# ============================================================================
# Setup Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "retraining.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Drift Detection
# ============================================================================
class DriftDetector:
    """Detect data drift using statistical tests."""
    
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Training data (baseline)
            threshold: P-value threshold for significance (default: 0.05)
        """
        self.reference_data = reference_data
        self.threshold = threshold
        logger.info(f"DriftDetector initialized with {len(reference_data)} reference samples")
    
    def detect_numeric_drift(self, new_data: pd.DataFrame, column: str) -> Dict:
        """
        Detect drift in numeric column using Kolmogorov-Smirnov test.
        
        Args:
            new_data: New data to check
            column: Column name
            
        Returns:
            Dict with test statistic and p-value
        """
        ref_values = self.reference_data[column].dropna().values
        new_values = new_data[column].dropna().values
        
        if len(ref_values) == 0 or len(new_values) == 0:
            return {"drifted": False, "reason": "insufficient data"}
        
        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(ref_values, new_values)
        
        drifted = p_value < self.threshold
        
        return {
            "column": column,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drifted": drifted,
            "ref_mean": float(ref_values.mean()),
            "new_mean": float(new_values.mean()),
            "ref_std": float(ref_values.std()),
            "new_std": float(new_values.std())
        }
    
    def detect_categorical_drift(self, new_data: pd.DataFrame, column: str) -> Dict:
        """
        Detect drift in categorical column using Chi-square test.
        
        Args:
            new_data: New data to check
            column: Column name
            
        Returns:
            Dict with test statistic and p-value
        """
        ref_counts = self.reference_data[column].value_counts(normalize=True)
        new_counts = new_data[column].value_counts(normalize=True)
        
        # Align indices
        all_categories = set(ref_counts.index) | set(new_counts.index)
        ref_counts = ref_counts.reindex(all_categories, fill_value=0.0001)
        new_counts = new_counts.reindex(all_categories, fill_value=0.0001)
        
        # Chi-square test
        statistic, p_value = stats.chisquare(new_counts.values, ref_counts.values)
        
        drifted = p_value < self.threshold
        
        return {
            "column": column,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drifted": drifted
        }
    
    def check_all_drift(self, new_data: pd.DataFrame) -> Dict:
        """
        Check drift across all columns.
        
        Args:
            new_data: New data to check
            
        Returns:
            Summary of drift detection
        """
        logger.info(f"Checking drift on {len(new_data)} new samples")
        
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(new_data),
            "numeric_drift": [],
            "categorical_drift": [],
            "overall_drift": False
        }
        
        # Check numeric columns
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in new_data.columns:
                result = self.detect_numeric_drift(new_data, col)
                drift_results["numeric_drift"].append(result)
                if result["drifted"]:
                    drift_results["overall_drift"] = True
                    logger.warning(f"Drift detected in {col}: p-value={result['p_value']:.4f}")
        
        # Check categorical columns
        categorical_cols = self.reference_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in new_data.columns:
                result = self.detect_categorical_drift(new_data, col)
                drift_results["categorical_drift"].append(result)
                if result["drifted"]:
                    drift_results["overall_drift"] = True
                    logger.warning(f"Drift detected in {col}: p-value={result['p_value']:.4f}")
        
        return drift_results


# ============================================================================
# Model Versioning
# ============================================================================
class ModelRegistry:
    """Manage model versions and metadata."""
    
    def __init__(self, registry_dir: Path = MODELS_DIR):
        """Initialize model registry."""
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "registry.json"
        
        # Load existing registry
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {"models": []}
        
        logger.info(f"ModelRegistry initialized at {self.registry_dir}")
    
    def register_model(self, model_path: Path, metrics: Dict, metadata: Dict = None) -> str:
        """
        Register a new model version.
        
        Args:
            model_path: Path to model file
            metrics: Model performance metrics
            metadata: Additional metadata
            
        Returns:
            Model version ID
        """
        version_id = f"v{len(self.registry['models']) + 1}"
        timestamp = datetime.now().isoformat()
        
        model_record = {
            "version_id": version_id,
            "timestamp": timestamp,
            "model_path": str(model_path),
            "metrics": metrics,
            "metadata": metadata or {},
            "status": "active"
        }
        
        self.registry["models"].append(model_record)
        self._save_registry()
        
        logger.info(f"Registered model {version_id} with AUC={metrics.get('auc', 0):.4f}")
        return version_id
    
    def get_latest_model(self) -> Dict:
        """Get the latest active model."""
        active_models = [m for m in self.registry["models"] if m.get("status") == "active"]
        if active_models:
            return active_models[-1]
        return None
    
    def promote_model(self, version_id: str) -> bool:
        """Promote a model to production."""
        for model in self.registry["models"]:
            if model["version_id"] == version_id:
                model["status"] = "production"
                self._save_registry()
                logger.info(f"Promoted {version_id} to production")
                return True
        return False
    
    def _save_registry(self):
        """Save registry to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.registry, f, indent=2)


# ============================================================================
# Retraining Pipeline
# ============================================================================
def check_retraining_needed(new_data: pd.DataFrame, reference_data: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Determine if retraining is needed based on:
    1. Data drift
    2. New data volume
    3. Time since last training
    
    Args:
        new_data: Recent data
        reference_data: Training data
        
    Returns:
        Tuple of (should_retrain, analysis_results)
    """
    logger.info("Checking if retraining is needed...")
    
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "new_samples": len(new_data),
        "criteria": {}
    }
    
    should_retrain = False
    
    # Criterion 1: Data Drift
    logger.info("Checking for data drift...")
    drift_detector = DriftDetector(reference_data)
    drift_results = drift_detector.check_all_drift(new_data)
    
    analysis["criteria"]["drift"] = {
        "detected": drift_results["overall_drift"],
        "details": drift_results
    }
    
    if drift_results["overall_drift"]:
        should_retrain = True
        logger.warning("Data drift detected - retraining recommended")
    
    # Criterion 2: New Data Volume
    new_samples_threshold = len(reference_data) * 0.1  # 10% of training data
    analysis["criteria"]["volume"] = {
        "new_samples": len(new_data),
        "threshold": new_samples_threshold,
        "threshold_met": len(new_data) >= new_samples_threshold
    }
    
    if len(new_data) >= new_samples_threshold:
        should_retrain = True
        logger.info(f"New data volume threshold met ({len(new_data)} samples)")
    
    # Criterion 3: Time Since Last Training
    # (would compare with last training timestamp)
    days_since_training = 7  # Simulated
    retrain_interval_days = 7
    
    analysis["criteria"]["time"] = {
        "days_since_training": days_since_training,
        "retrain_interval_days": retrain_interval_days,
        "threshold_met": days_since_training >= retrain_interval_days
    }
    
    if days_since_training >= retrain_interval_days:
        should_retrain = True
        logger.info(f"Scheduled retraining interval reached ({days_since_training} days)")
    
    return should_retrain, analysis


def run_retraining() -> Dict:
    """
    Execute full retraining pipeline.
    
    Returns:
        Training results with metrics
    """
    logger.info("="*70)
    logger.info("STARTING AUTOMATED RETRAINING PIPELINE")
    logger.info("="*70)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "status": "failed",
        "error": None,
        "model_version": None
    }
    
    try:
        # ====================================================================
        # Step 1: Load New Data
        # ====================================================================
        logger.info("[STEP 1/5] Loading new data...")
        
        # Simulate fetching new data (in production: from database, S3, etc.)
        raw_data_file = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        
        if not raw_data_file.exists():
            logger.error(f"Data file not found: {raw_data_file}")
            raise FileNotFoundError(f"Dataset not found at {raw_data_file}")
        
        df_all = pd.read_csv(raw_data_file)
        
        # Simulate: last 500 records are "new"
        df_reference = df_all.iloc[:-500].copy()
        df_new = df_all.iloc[-500:].copy()
        
        logger.info(f"Loaded {len(df_reference)} reference samples, {len(df_new)} new samples")
        
        # ====================================================================
        # Step 2: Check Drift
        # ====================================================================
        logger.info("[STEP 2/5] Checking for data drift...")
        
        should_retrain, drift_analysis = check_retraining_needed(df_new, df_reference)
        results["drift_analysis"] = drift_analysis
        
        if not should_retrain:
            logger.info("No retraining needed based on drift analysis")
            results["status"] = "skipped"
            return results
        
        logger.info("Retraining triggered based on drift analysis")
        
        # ====================================================================
        # Step 3: Prepare Data & Train
        # ====================================================================
        logger.info("[STEP 3/5] Training new model...")
        
        # Combine reference + new data for training
        df_combined = pd.concat([df_reference, df_new], ignore_index=True)
        
        # Write the combined dataset to a temporary CSV so the existing data pipeline can process it
        data_pipeline = DataPipeline()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temp_file:
            temp_file_path = Path(temp_file.name)
            df_combined.to_csv(temp_file_path, index=False)

        try:
            X_train, X_test, y_train, y_test = data_pipeline.process(
                temp_file_path, save=False
            )
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()

        comparator = ModelComparison(cv_folds=CV_FOLDS)
        baseline_scores = comparator.quick_baseline(X_train, y_train, X_test, y_test)
        tuned_results = comparator.tune_top_models(X_train, y_train, X_test, y_test, top_k=2)

        if not tuned_results:
            raise RuntimeError("No tuned models were generated during retraining.")

        best_model_name = max(tuned_results.items(), key=lambda x: x[1]["test_auc"])[0]
        best_model = tuned_results[best_model_name]["model"]
        best_auc = tuned_results[best_model_name]["test_auc"]

        logger.info(f"New model trained: {best_model_name}, AUC={best_auc:.4f}")

        results["model_name"] = best_model_name
        results["metrics"] = {
            "auc": best_auc,
            "precision": tuned_results[best_model_name]["precision"],
            "recall": tuned_results[best_model_name]["recall"],
            "f1": tuned_results[best_model_name]["f1"]
        }
        
        # ====================================================================
        # Step 4: Compare with Production Model
        # ====================================================================
        logger.info("[STEP 4/5] Comparing with production model...")
        
        # Load current production model
        production_model_path = MODELS_DIR / "production_model.pkl"
        
        if production_model_path.exists():
            with open(production_model_path, 'rb') as f:
                production_model = pickle.load(f)
            
            y_pred_prod = production_model.predict_proba(X_test)[:, 1]
            prod_auc = roc_auc_score(y_test, y_pred_prod)
            
            logger.info(f"Production AUC: {prod_auc:.4f}, New AUC: {best_auc:.4f}")
            
            # Only deploy if new model is better
            improvement = best_auc - prod_auc
            results["improvement"] = improvement
            
            if improvement > 0.01:  # Minimum 1% improvement
                logger.info(f"New model is better by {improvement:.4f}")
                deploy_new = True
            else:
                logger.warning(f"New model not significantly better ({improvement:.4f})")
                deploy_new = False
        else:
            logger.info("No production model found - deploying new model")
            deploy_new = True
        
        # ====================================================================
        # Step 5: Register & Deploy Model
        # ====================================================================
        logger.info("[STEP 5/5] Registering and deploying model...")
        
        if deploy_new:
            # Save new model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_model_path = MODELS_DIR / f"{best_model_name}_{timestamp}.pkl"
            
            with open(versioned_model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            # Update production model symlink
            if production_model_path.exists():
                production_model_path.unlink()
            
            with open(production_model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            # Register in model registry
            registry = ModelRegistry()
            version_id = registry.register_model(
                versioned_model_path,
                results["metrics"],
                {
                    "model_name": best_model_name,
                    "training_samples": len(df_combined),
                    "test_auc_improvement": results.get("improvement", 0)
                }
            )
            
            results["model_version"] = version_id
            results["status"] = "success"
            
            logger.info(f"✓ New model deployed as {version_id}")
            
            # Send notification
            _notify_deployment(results)
        else:
            results["status"] = "trained_but_not_deployed"
            logger.info("New model trained but not deployed (not better than production)")
        
        logger.info("="*70)
        logger.info("RETRAINING COMPLETE ✓")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}", exc_info=True)
        results["error"] = str(e)
        results["status"] = "failed"
    
    return results


def _notify_deployment(results: Dict):
    """Send notification of deployment (Slack, email, etc.)."""
    # Example: Slack notification
    webhook_url = None  # Set via environment variable
    
    if webhook_url:
        message = {
            "text": "🚀 Model Deployment",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Model Deployed*\n" +
                               f"Version: {results.get('model_version')}\n" +
                               f"AUC: {results.get('metrics', {}).get('auc', 0):.4f}\n" +
                               f"Status: {results.get('status')}"
                    }
                }
            ]
        }
        
        try:
            requests.post(webhook_url, json=message, timeout=10)
            logger.info("Slack notification sent")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    results = run_retraining()
    
    # Log summary
    print("\n" + "="*70)
    print("RETRAINING SUMMARY")
    print("="*70)
    print(f"Status: {results['status']}")
    if results.get('model_version'):
        print(f"Model Version: {results['model_version']}")
    if results.get('metrics'):
        print(f"Metrics: AUC={results['metrics'].get('auc', 0):.4f}")
    if results.get('error'):
        print(f"Error: {results['error']}")
    print("="*70 + "\n")