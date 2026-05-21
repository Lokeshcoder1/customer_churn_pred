import os
from pathlib import Path
from typing import List

#Paths
PROJECT_ROOT=Path(__file__).parent.parent
DATA_DIR=PROJECT_ROOT/"data"
RAW_DATA_DIR=DATA_DIR/"raw"
PROCESSED_DATA_DIR=DATA_DIR/"processed"
MODELS_DIR=PROJECT_ROOT/"models"
ARTIFACTS_DIR=PROJECT_ROOT/"artifacts"
LOGS_DIR=PROJECT_ROOT/"logs"

#create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
                  ARTIFACTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True,exist_ok=True)

#Data Parameters
TEST_SIZE=0.2
RANDOM_STATE=42
TARGET_COLUMN="Churn"
CHURN_VALUE="Yes"


#Feature columns
NUMERIC_FEATURES=["tenure","MonthlyCharges","TotalCharges"]
CATEGORICAL_FEATURES=[
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group"
]

#Features to drop
FEATURES_TO_DROP=["customerID"]

#All Features
ALL_FEATURES=NUMERIC_FEATURES+CATEGORICAL_FEATURES

# MODEL HYPERPARAMETERS
XGBOOST_HYPERPARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
    "verbosity": 0
}

LIGHTGBM_HYPERPARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "random_state": RANDOM_STATE,
    "verbose": -1
}

RANDOM_FOREST_HYPERPARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

LOGISTIC_REGRESSION_HYPERPARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": RANDOM_STATE
}

# MODEL COMPARISON SETTINGS
CV_FOLDS = 5
CHURN_PREDICTION_THRESHOLD = 0.5

# API SETTINGS
API_TITLE = "Customer Churn Prediction API"
API_DESCRIPTION = "Predicts customer churn risk for telecom companies"
API_VERSION = "1.0.0"
MODEL_ARTIFACT_PATH = MODELS_DIR / "production_model.pkl"

# LOGGING CONFIGURATION
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"