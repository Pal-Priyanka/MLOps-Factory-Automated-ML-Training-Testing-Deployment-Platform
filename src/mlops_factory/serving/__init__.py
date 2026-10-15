"""Inference service, REST API endpoints, and Pydantic schemas."""

from mlops_factory.serving.app import create_app
from mlops_factory.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    PredictionRequest,
    PredictionResponse,
)

__all__ = [
    "create_app",
    "CustomerFeatures",
    "PredictionRequest",
    "PredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
]
