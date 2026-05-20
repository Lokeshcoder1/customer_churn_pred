                       # XGBoost Hyperparameter Space

# XGBoost is the primary model for this task (best on tabular data)
XGBOOST_PARAM_GRID = {
    "model__n_estimators": [50, 100, 150],      # Number of boosting rounds
    "model__max_depth": [3, 5, 7],              # Tree depth (deeper = more complex)
    "model__learning_rate": [0.01, 0.1, 0.2],  # Shrinkage factor (lower = slower but stable)
    "model__subsample": [0.7, 0.9],             # Fraction of samples per iteration
    "model__colsample_bytree": [0.7, 0.9]      # Fraction of features per tree
}

                    # LightGBM Hyperparameter Space

# LightGBM: faster than XGBoost, often similar performance
LIGHTGBM_PARAM_GRID = {
    "model__n_estimators": [50, 100, 150],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.1, 0.2],
    "model__num_leaves": [15, 31, 63],          # LightGBM uses leaves instead of depth
    "model__feature_fraction": [0.7, 0.9]       # Feature sampling ratio
}


                # Random Forest Hyperparameter Space

# Random Forest: simpler baseline, faster training
RANDOM_FOREST_PARAM_GRID = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [5, 10, 15],
    "model__min_samples_split": [5, 10],        # Min samples to split a node
    "model__min_samples_leaf": [2, 4]           # Min samples in leaf node
}
# Total combinations: 3 * 3 * 2 * 2 = 36


                # Logistic Regression Hyperparameter Space

# Logistic Regression: linear baseline
LOGISTIC_REGRESSION_PARAM_GRID = {
    "model__C": [0.001, 0.01, 0.1, 1.0, 10.0],  # Inverse of regularization strength
    "model__penalty": ["l2"]                     # L2 regularization
}
