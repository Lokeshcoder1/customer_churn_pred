"""
Pydantic Schemas for API
Defines request and response formats with validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ChurnPredictionRequest(BaseModel):
    """
    Single customer prediction request.

    Validates input data types and ranges.
    """

    tenure: float = Field(..., gt=0, description="Number of months as customer (must be > 0)")
    monthly_charges: float = Field(..., gt=0, description="Monthly service charges (must be > 0)")
    total_charges: float = Field(..., ge=0, description="Total charges to date (must be >= 0)")

    # Contract features
    contract: str = Field(..., description="Contract type: Month-to-month, One year, Two year")

    # Service features
    internet_service: str = Field(..., description="Type of internet service: DSL, Fiber optic, No")
    online_security: str = Field(..., description="Online security service: Yes, No, No internet service")
    online_backup: str = Field(..., description="Online backup: Yes, No, No internet service")
    device_protection: str = Field(..., description="Device protection: Yes, No, No internet service")
    tech_support: str = Field(..., description="Tech support: Yes, No, No internet service")

    # Demographic features
    senior_citizen: int = Field(..., ge=0, le=1, description="Is senior citizen: 0=No, 1=Yes")
    partner: str = Field(..., description="Has partner: Yes, No")
    dependents: str = Field(..., description="Has dependents: Yes, No")

    # Other features
    gender: str = Field(..., description="Gender: Male, Female")
    phone_service: str = Field(..., description="Phone service: Yes, No")
    streaming_tv: str = Field(..., description="Streaming TV: Yes, No, No internet service")
    streaming_movies: str = Field(..., description="Streaming movies: Yes, No, No internet service")
    paperless_billing: str = Field(..., description="Paperless billing: Yes, No")
    payment_method: str = Field(...,
                                description="Payment method: Bank transfer, Credit card, Electronic check, Mailed check")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
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
        }


class ChurnPredictionResponse(BaseModel):
    """
    Single prediction response.
    """

    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn (0-1)")
    churn_prediction: bool = Field(..., description="Will customer churn? True/False")
    confidence: str = Field(..., description="Model confidence: high, medium, low")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "churn_probability": 0.7234,
                "churn_prediction": True,
                "confidence": "high"
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Batch prediction request (multiple customers).
    """

    customers: List[ChurnPredictionRequest] = Field(..., description="List of customers to predict")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
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
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Batch prediction response.
    """

    predictions: List[ChurnPredictionResponse] = Field(..., description="List of predictions")
    count: int = Field(..., description="Number of predictions made")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "churn_probability": 0.7234,
                        "churn_prediction": True,
                        "confidence": "high"
                    }
                ],
                "count": 1
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response.
    """

    status: str = Field(..., description="Service status: healthy, unhealthy")
    model_loaded: bool = Field(..., description="Is model loaded?")
    message: Optional[str] = Field(None, description="Additional message")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "message": "API running normally"
            }
        }


class ErrorResponse(BaseModel):
    """
    Error response.
    """

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error detail")
    status_code: int = Field(..., description="HTTP status code")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "tenure must be greater than 0",
                "status_code": 422
            }
        }