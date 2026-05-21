"""
Main Training Script
Orchestrates the entire ML pipeline: data → comparison → tuning → saving
"""

import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RAW_DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, LOG_LEVEL, LOG_FORMAT
from src.data_pipeline import DataPipeline
from src.model_comparison import ModelComparison
from src.pipeline import MLPipeline
from sklearn.pipeline import Pipeline

# ============================================================================
# Setup Logging
# ============================================================================
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / "logs" / "training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main training pipeline.

    Flow:
    1. Load and process data
    2. Compare models (baseline + tuning)
    3. Save best model
    4. Generate reports and visualizations
    """

    logger.info("=" * 80)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 80)

    try:
        # ====================================================================
        # Step 1: Data Processing
        # ====================================================================
        logger.info("\n[STEP 1/4] Data Processing")
        logger.info("-" * 80)

        data_pipeline = DataPipeline()

        # Check if data file exists
        data_file = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            logger.info("\nPlease download the Telco Customer Churn dataset from:")
            logger.info("https://www.kaggle.com/blastchar/telco-customer-churn")
            logger.info(f"\nAnd place it at: {data_file}")
            return

        # Process data
        X_train, X_test, y_train, y_test = data_pipeline.process(data_file, save=True)

        logger.info(f"[ok] Data processing complete")
        logger.info(f"  Training set: {X_train.shape}")
        logger.info(f"  Test set: {X_test.shape}")

        # ====================================================================
        # Step 2: Model Comparison (Baseline)
        # ====================================================================
        logger.info("\n[STEP 2/4] Model Comparison & Baseline")
        logger.info("-" * 80)

        comparator = ModelComparison(cv_folds=5)

        # Quick baseline with all 4 models
        baseline_scores = comparator.quick_baseline(X_train, y_train, X_test, y_test)

        logger.info(f"[ok] Baseline comparison complete")
        logger.info(f"  Models compared: {len(baseline_scores)}")

        # ====================================================================
        # Step 3: Hyperparameter Tuning
        # ====================================================================
        logger.info("\n[STEP 3/4] Hyperparameter Tuning")
        logger.info("-" * 80)

        # Tune top 2 models
        tuned_results = comparator.tune_top_models(X_train, y_train, X_test, y_test, top_k=2)

        logger.info(f"[ok] Hyperparameter tuning complete")
        logger.info(f"  Models tuned: {len(tuned_results)}")

        # ====================================================================
        # Step 4: Save Results & Reports
        # ====================================================================
        logger.info("\n[STEP 4/4] Save Results & Reports")
        logger.info("-" * 80)

        # Generate visualizations
        comparator.plot_comparisons(X_test, y_test)
        logger.info("[ok] Generated ROC curve plots")

        # Select best model
        best_model_name = max(tuned_results.items(), key=lambda x: x[1]["test_auc"])[0]
        best_model_sklearn = tuned_results[best_model_name]["model"]
        best_auc = tuned_results[best_model_name]["test_auc"]

        logger.info(f"\n{'=' * 80}")
        logger.info(f"BEST MODEL: {best_model_name}")
        logger.info(f"Test AUC: {best_auc:.4f}")
        logger.info(f"{'=' * 80}")

        # Wrap in MLPipeline and save
        ml_pipeline = MLPipeline()
        ml_pipeline.set_pipeline(best_model_sklearn)
        ml_pipeline.save()

        logger.info(f"[ok] Saved model for production")

        # Now generate report (safe to fail here)
        try:
            comparator.save_report(y_test)
            logger.info("[ok] Generated comparison report")
        except Exception as e:
            logger.warning(f"Could not save report: {e}")

        # ====================================================================
        # Summary
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE ✓")
        logger.info("=" * 80)

        logger.info(f"\nKey Results:")
        logger.info(f"  Best Model: {best_model_name}")
        logger.info(f"  Test AUC: {best_auc:.4f}")
        logger.info(f"  Test Precision: {tuned_results[best_model_name]['precision']:.4f}")
        logger.info(f"  Test Recall: {tuned_results[best_model_name]['recall']:.4f}")

        logger.info(f"\nOutput Files:")
        logger.info(f"  Report: {ARTIFACTS_DIR}/model_comparison_report.txt")
        logger.info(f"  Plots: {ARTIFACTS_DIR}/model_comparison_roc_curves.png")
        logger.info(f"  Model: {MODELS_DIR}/production_model.pkl")

        logger.info(f"\nNext Steps:")
        logger.info(f"  1. Review report: cat logs/model_comparison_report.txt")
        logger.info(f"  2. Start API: uvicorn src.api:app --reload")
        logger.info(f"  3. Test predictions at: http://localhost:8000/docs")

        logger.info("\n" + "=" * 80 + "\n")

        return True

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)