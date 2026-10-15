"""FastAPI inference service exposing health, metadata, prediction, and telemetry endpoints."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from mlops_factory import __version__
from mlops_factory.config import DEFAULT_CONFIG, PipelineConfig
from mlops_factory.features.preprocessor import clean_inference_features
from mlops_factory.monitoring.tracker import metrics_tracker
from mlops_factory.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from mlops_factory.training.train import load_and_validate_model_artifact

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("mlops_factory.serving")


class ModelContainer:
    """Holds active model pipeline and metadata in application state."""
    pipeline: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    is_ready: bool = False
    init_error: Optional[str] = None


model_container = ModelContainer()


def assign_risk_tier_and_action(churn_prob: float) -> tuple[str, str]:
    """Maps churn probability to business risk tiers and proactive retention workflows."""
    if churn_prob >= 0.70:
        return "High", "Immediate outreach: assign dedicated retention specialist with competitive tariff match."
    elif churn_prob >= 0.40:
        return "Medium", "Proactive engagement: offer contract renewal discount or bundled service upgrade."
    else:
        return "Low", "Standard account maintenance; no immediate intervention required."


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    artifacts_dir = Path("artifacts")
    logger.info("Initializing inference service. Verifying model artifacts at '%s'...", artifacts_dir)

    try:
        pipeline, metadata = load_and_validate_model_artifact(artifacts_dir=artifacts_dir)
        model_container.pipeline = pipeline
        model_container.metadata = metadata
        model_container.is_ready = True
        logger.info(
            "Model '%s' successfully loaded and verified (SHA256: %s). Service is READY.",
            metadata.get("model_name"),
            metadata.get("model_sha256", "N/A")[:12],
        )
    except Exception as exc:
        model_container.is_ready = False
        model_container.init_error = str(exc)
        logger.warning(
            "Model artifact could not be loaded at startup: %s. "
            "Endpoints requiring model will return 503 until training is executed.",
            exc,
        )

    yield

    logger.info("Shutting down inference service.")


def create_app(config: PipelineConfig = DEFAULT_CONFIG) -> FastAPI:
    """Factory function for FastAPI application."""
    app = FastAPI(
        title=config.api_title,
        version=config.api_version,
        description="Production REST API for customer churn risk prediction, operational monitoring, and telemetry.",
        lifespan=lifespan,
    )

    # Middleware for latency and structured request tracking
    @app.middleware("http")
    async def request_telemetry_middleware(request: Request, call_next):
        req_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

        # Safe logging without leaking customer payload
        if request.url.path not in ["/health", "/ready", "/metrics", "/metrics/prometheus"]:
            logger.info(
                "[%s] %s %s -> status=%d latency=%.2fms",
                req_id,
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )

        return response

    # Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        metrics_tracker.record_error()
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "message": "Supplied input payload does not match expected schema contract.",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        metrics_tracker.record_error()
        logger.error("Unhandled server exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred while processing the request.",
            },
        )

    # Liveness check
    @app.get("/health", response_model=HealthResponse, tags=["Operations"])
    async def health_check():
        """Liveness check for orchestration tools."""
        snap = metrics_tracker.get_metrics_snapshot()
        return HealthResponse(
            status="healthy",
            version=__version__,
            model_loaded=model_container.is_ready,
            uptime_seconds=snap.uptime_seconds,
        )

    # Readiness check
    @app.get("/ready", tags=["Operations"])
    async def readiness_check():
        """Readiness check. Returns 200 if model artifact is ready, 503 otherwise."""
        if not model_container.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service unready: {model_container.init_error or 'Model artifact not loaded'}",
            )
        return {"status": "ready", "model_name": model_container.metadata.get("model_name")}

    # Metadata endpoint
    @app.get("/metadata", response_model=MetadataResponse, tags=["Model Governance"])
    async def get_metadata():
        """Returns provenance, feature definitions, and quality gate outcomes for the active model."""
        if not model_container.is_ready or model_container.metadata is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model metadata is unavailable because no model artifact has been loaded.",
            )

        meta = model_container.metadata
        return MetadataResponse(
            model_name=meta.get("model_name", "Unknown"),
            trained_at_utc=meta.get("trained_at_utc", "Unknown"),
            model_sha256=meta.get("model_sha256", "Unknown"),
            total_features=meta.get("features", {}).get("total_features", 0),
            scikit_learn_version=meta.get("environment", {}).get("scikit_learn_version", "Unknown"),
            quality_gate_passed=meta.get("quality_gates", {}).get("passed", False),
            test_metrics=meta.get("quality_gates", {}).get("checks", []),
        )

    # Single prediction endpoint
    @app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
    async def predict(request_payload: PredictionRequest):
        """
        Executes real-time churn risk inference on a customer feature profile.
        """
        if not model_container.is_ready or model_container.pipeline is None:
            metrics_tracker.record_error()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready to serve predictions. Ensure training artifact is generated.",
            )

        start_time = time.perf_counter()
        try:
            # Format row matching training DataFrame structure
            row_dict = request_payload.customer.to_dataframe_row()
            df_in = pd.DataFrame([row_dict])

            # Clean and align features using preprocessor rules
            df_clean = clean_inference_features(df_in, config=config)

            # Predict
            pipeline = model_container.pipeline
            prediction_int = int(pipeline.predict(df_clean)[0])

            if hasattr(pipeline, "predict_proba"):
                churn_prob = float(pipeline.predict_proba(df_clean)[0][1])
            else:
                churn_prob = float(prediction_int)

            churn_prob = round(churn_prob, 4)
            churn_bool = bool(prediction_int == 1)

            risk_tier, action = assign_risk_tier_and_action(churn_prob)
            duration_s = time.perf_counter() - start_time
            duration_ms = round(duration_s * 1000.0, 2)

            # Record telemetry
            metrics_tracker.record_prediction(prediction_int, duration_s)

            return PredictionResponse(
                churn_prediction=churn_bool,
                churn_probability=churn_prob,
                risk_tier=risk_tier,
                recommended_action=action,
                model_name=model_container.metadata.get("model_name", "ProductionModel"),
                model_version=model_container.metadata.get("model_sha256", "v1.0")[:8],
                latency_ms=duration_ms,
            )
        except Exception as exc:
            metrics_tracker.record_error()
            logger.error("Inference execution failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred during model prediction.",
            )

    # Batch prediction endpoint
    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
    async def predict_batch(request_payload: BatchPredictionRequest):
        """
        Executes batch inference over a collection of customer profiles (up to 1,000 items).
        """
        if not model_container.is_ready or model_container.pipeline is None:
            metrics_tracker.record_error()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready to serve predictions.",
            )

        start_time = time.perf_counter()
        try:
            rows = [c.to_dataframe_row() for c in request_payload.customers]
            df_in = pd.DataFrame(rows)
            df_clean = clean_inference_features(df_in, config=config)

            pipeline = model_container.pipeline
            predictions = pipeline.predict(df_clean)

            if hasattr(pipeline, "predict_proba"):
                probabilities = pipeline.predict_proba(df_clean)[:, 1]
            else:
                probabilities = predictions.astype(float)

            total_duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            per_item_ms = round(total_duration_ms / max(1, len(rows)), 3)

            results: List[PredictionResponse] = []
            churn_count = 0
            model_name = model_container.metadata.get("model_name", "ProductionModel")
            model_ver = model_container.metadata.get("model_sha256", "v1.0")[:8]

            for pred_int, prob in zip(predictions, probabilities):
                p_int = int(pred_int)
                p_prob = round(float(prob), 4)
                if p_int == 1:
                    churn_count += 1
                risk_tier, action = assign_risk_tier_and_action(p_prob)
                metrics_tracker.record_prediction(p_int, per_item_ms / 1000.0)

                results.append(
                    PredictionResponse(
                        churn_prediction=bool(p_int == 1),
                        churn_probability=p_prob,
                        risk_tier=risk_tier,
                        recommended_action=action,
                        model_name=model_name,
                        model_version=model_ver,
                        latency_ms=per_item_ms,
                    )
                )

            return BatchPredictionResponse(
                total_processed=len(results),
                predicted_churners_count=churn_count,
                predictions=results,
                batch_latency_ms=total_duration_ms,
            )
        except Exception as exc:
            metrics_tracker.record_error()
            logger.error("Batch inference failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred during batch model prediction.",
            )

    # Operational metrics (JSON)
    @app.get("/metrics", tags=["Operations"])
    async def get_metrics():
        """Returns JSON summary of operational metrics (latencies, counts, predictions)."""
        return metrics_tracker.get_metrics_snapshot().to_dict()

    # Prometheus exposition endpoint
    @app.get("/metrics/prometheus", response_class=PlainTextResponse, tags=["Operations"])
    async def get_prometheus_metrics():
        """Returns Prometheus text exposition format for telemetry scrapers."""
        return metrics_tracker.to_prometheus_format()

    return app


app = create_app()
