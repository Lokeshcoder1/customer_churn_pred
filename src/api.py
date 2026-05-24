"""
FastAPI Application
Main API server for churn predictions.

Run with: uvicorn src.api:app --reload
Visit: http://localhost:8000/docs for interactive documentation
"""

import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .schemas import (
    ChurnPredictionRequest, ChurnPredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ErrorResponse
)
from .pipeline import MLPipeline
from .data_pipeline import DataPipeline
from .config import API_TITLE, API_DESCRIPTION, API_VERSION, MODEL_ARTIFACT_PATH

# ============================================================================
# Setup Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# Global State
# ============================================================================
ml_pipeline = MLPipeline()
data_pipeline = DataPipeline()

RAW_TO_API_FIELD_MAP = {
    "tenure": "tenure",
    "monthlycharges": "monthly_charges",
    "totalcharges": "total_charges",
    "contract": "contract",
    "internetservice": "internet_service",
    "onlinesecurity": "online_security",
    "onlinebackup": "online_backup",
    "deviceprotection": "device_protection",
    "techsupport": "tech_support",
    "seniorcitizen": "senior_citizen",
    "partner": "partner",
    "dependents": "dependents",
    "gender": "gender",
    "phoneservice": "phone_service",
    "multiplelines": "multiple_lines",
    "streamingtv": "streaming_tv",
    "streamingmovies": "streaming_movies",
    "paperlessbilling": "paperless_billing",
    "paymentmethod": "payment_method",
}

def normalize_customer_keys(customer: dict) -> dict:
    normalized = {}
    for key, value in customer.items():
        clean_key = key.replace("_", "").replace(" ", "").lower()
        normalized_key = RAW_TO_API_FIELD_MAP.get(clean_key, key)
        normalized[normalized_key] = value
    return normalized


# ============================================================================
# Lifespan Events
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model on startup, clean up on shutdown.

    This is the recommended way in FastAPI to manage application lifecycle.
    """
    # Startup: Load model
    logger.info("Starting up API...")
    try:
        # load preprocessing artifacts first to align features
        try:
            data_pipeline.load_pipeline()
            logger.info("[ok] Data pipeline artifacts loaded")
        except Exception:
            logger.warning("No preprocessing artifacts found or failed to load; continuing")

        ml_pipeline.load(MODEL_ARTIFACT_PATH)
        logger.info("[ok] Model loaded successfully")
    except FileNotFoundError:
        logger.error("Model artifact not found. Please train a model first.")
        raise RuntimeError(
            f"Model not found at {MODEL_ARTIFACT_PATH}. "
            "Run training script first: python src/train.py"
        )
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

    yield  # Application is running

    # Shutdown: Clean up
    logger.info("Shutting down API...")


# ============================================================================
# Create FastAPI App
# ============================================================================
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan
)


# ============================================================================
# Exception Handlers
# ============================================================================
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "ValueError",
            "detail": str(exc),
            "status_code": 400
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred. Check server logs.",
            "status_code": 500
        }
    )


# ============================================================================
# Health Check Endpoint
# ============================================================================
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check",
    description="Check if API is running and model is loaded"
)
async def health_check():
    """
    Liveness probe for deployment (Docker, Kubernetes, etc).

    This endpoint is called by orchestration platforms to ensure the API is running.
    """
    try:
        model_loaded = ml_pipeline.pipeline is not None
        return HealthResponse(
            status="healthy",
            model_loaded=model_loaded,
            message="API running normally"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            message=f"Error: {str(e)}"
        )


# ============================================================================
# Single Prediction Endpoint
# ============================================================================
@app.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    tags=["Predictions"],
    summary="Predict Churn (Single Customer)",
    description="Predict whether a single customer will churn",
    responses={
        200: {"description": "Prediction successful"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        500: {"description": "Server error", "model": ErrorResponse}
    }
)
async def predict_single(request: dict = Body(...)):
    """
    Predict churn for a single customer.

    **Request Body:**
    - tenure: Number of months as customer (must be > 0)
    - monthly_charges: Monthly service charges (must be > 0)
    - total_charges: Total charges to date (must be >= 0)
    - Other features: See schema below

    **Response:**
    - churn_probability: Probability of churn (0-1)
    - churn_prediction: Will customer churn? (True/False)
    - confidence: Model confidence level (high/medium/low)

    **Example:**
    ```json
    {
        "tenure": 24,
        "monthly_charges": 65.5,
        "total_charges": 1570.0,
        "contract": "Two year",
        ...
    }
    ```
    """
    try:
        # Normalize raw dataset field names and validate payload
        features_dict = normalize_customer_keys(request)
        request_model = ChurnPredictionRequest.model_validate(features_dict)
        features_dict = request_model.model_dump()

        # Convert to DataFrame
        import pandas as pd
        X = pd.DataFrame([features_dict])

        # Prepare for prediction: map, encode, scale, and align
        try:
            X_prepared = data_pipeline.prepare_for_prediction(X)
        except Exception as e:
            raise ValueError(f"Preprocessing failed: {e}")

        # Get prediction with confidence
        results = ml_pipeline.predict_with_confidence(X_prepared)

        churn_proba = float(results["probabilities"][0])
        churn_pred = bool(results["predictions"][0])
        confidence = str(results["confidence"][0])

        logger.info(f"Prediction made: churn_prob={churn_proba:.4f}, confidence={confidence}")

        return ChurnPredictionResponse(
            churn_probability=round(churn_proba, 4),
            churn_prediction=churn_pred,
            confidence=confidence
        )

    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.errors()
        )
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================================
# Batch Prediction Endpoint
# ============================================================================
@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    summary="Predict Churn (Batch)",
    description="Predict churn for multiple customers at once",
    responses={
        200: {"description": "Predictions successful"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        500: {"description": "Server error", "model": ErrorResponse}
    }
)
async def predict_batch(request: dict = Body(...)):
    """
    Predict churn for multiple customers (batch mode).

    Useful for:
    - Daily batch scoring of customer base
    - Upstream jobs that need to score many customers
    - Data pipeline integration

    **Request Body:**
    ```json
    {
        "customers": [
            {
                "tenure": 24,
                "monthly_charges": 65.5,
                ...
            },
            {
                "tenure": 12,
                "monthly_charges": 45.0,
                ...
            }
        ]
    }
    ```

    **Response:**
    ```json
    {
        "predictions": [
            {
                "churn_probability": 0.7234,
                "churn_prediction": true,
                "confidence": "high"
            },
            {
                "churn_probability": 0.3421,
                "churn_prediction": false,
                "confidence": "high"
            }
        ],
        "count": 2
    }
    ```
    """
    try:
        if "customers" not in request or not request["customers"]:
            raise ValueError("customers list cannot be empty")

        # Normalize and validate each record
        import pandas as pd
        features_dicts = []
        for customer in request["customers"]:
            normalized = normalize_customer_keys(customer)
            request_model = ChurnPredictionRequest.model_validate(normalized)
            features_dicts.append(request_model.model_dump())
        X = pd.DataFrame(features_dicts)

        # Prepare batch for prediction
        try:
            X_prepared = data_pipeline.prepare_for_prediction(X)
        except Exception as e:
            raise ValueError(f"Preprocessing failed: {e}")

        # Batch prediction
        results = ml_pipeline.predict_with_confidence(X_prepared)

        predictions = [
            ChurnPredictionResponse(
                churn_probability=round(float(proba), 4),
                churn_prediction=bool(pred),
                confidence=str(conf)
            )
            for proba, pred, conf in zip(
                results["probabilities"],
                results["predictions"],
                results["confidence"]
            )
        ]

        logger.info(f"Batch prediction completed for {len(predictions)} customers")

        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )

    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.errors()
        )
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# ============================================================================
# Model Info Endpoint
# ============================================================================
@app.get(
    "/info",
    tags=["System"],
    summary="Model Information",
    description="Get information about the loaded model"
)
async def model_info():
    """
    Get information about the loaded model.

    Returns:
    - model_version: Version of the model
    - status: Whether model is ready
    - info: Model architecture summary
    """
    try:
        if ml_pipeline.pipeline is None:
            return {
                "status": "no_model",
                "message": "No model loaded"
            }

        return {
            "status": "ready",
            "api_version": API_VERSION,
            "model_info": ml_pipeline.model_summary()
        }

    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# Root Endpoint
# ============================================================================
@app.get(
    "/",
    tags=["System"],
    summary="Welcome",
    description="Welcome to Churn Prediction API"
)
async def root():
    """
    Welcome endpoint. Directs to documentation.
    """
    return {
        "message": "Welcome to Customer Churn Prediction API",
        "version": API_VERSION,
        "docs": "/docs",
        "redoc": "/redoc"
    }


# ============================================================================
# Run Instructions
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("Starting Churn Prediction API")
    print("=" * 70)
    print("\nAPI will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    print("ReDoc at: http://localhost:8000/redoc")
    print("\n" + "=" * 70 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )