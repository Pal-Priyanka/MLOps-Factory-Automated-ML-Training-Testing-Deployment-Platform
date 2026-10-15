"""Operational monitoring tracker for request latencies, counts, errors, and prediction distributions."""

import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List
import numpy as np


@dataclass
class OperationalMetricsSnapshot:
    uptime_seconds: float
    total_requests: int
    successful_predictions: int
    failed_requests: int
    prediction_classes: Dict[str, int]
    latency_ms: Dict[str, float]
    recorded_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsTracker:
    """Thread-safe operational telemetry tracker for inference service."""

    def __init__(self, max_latency_samples: int = 5000):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_predictions = 0
        self.failed_requests = 0
        self.prediction_classes: Dict[str, int] = {"0": 0, "1": 0}
        self.latency_samples: List[float] = []
        self.max_latency_samples = max_latency_samples

    def record_prediction(self, prediction_class: int, latency_seconds: float) -> None:
        """Records a successful prediction and latency in milliseconds."""
        latency_ms = latency_seconds * 1000.0
        with self._lock:
            self.total_requests += 1
            self.successful_predictions += 1
            cls_key = str(prediction_class)
            self.prediction_classes[cls_key] = self.prediction_classes.get(cls_key, 0) + 1
            self.latency_samples.append(latency_ms)
            if len(self.latency_samples) > self.max_latency_samples:
                # Retain recent samples
                self.latency_samples = self.latency_samples[-self.max_latency_samples:]

    def record_error(self) -> None:
        """Increments error counter."""
        with self._lock:
            self.total_requests += 1
            self.failed_requests += 1

    def get_metrics_snapshot(self) -> OperationalMetricsSnapshot:
        """Computes summary statistics over captured latencies and requests."""
        with self._lock:
            uptime = round(time.time() - self.start_time, 2)
            total = self.total_requests
            success = self.successful_predictions
            failed = self.failed_requests
            pred_classes = dict(self.prediction_classes)
            latencies = list(self.latency_samples)

        if latencies:
            lat_summary = {
                "count": len(latencies),
                "mean_ms": round(float(np.mean(latencies)), 2),
                "min_ms": round(float(np.min(latencies)), 2),
                "max_ms": round(float(np.max(latencies)), 2),
                "p50_ms": round(float(np.percentile(latencies, 50)), 2),
                "p95_ms": round(float(np.percentile(latencies, 95)), 2),
                "p99_ms": round(float(np.percentile(latencies, 99)), 2),
            }
        else:
            lat_summary = {
                "count": 0,
                "mean_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
            }

        return OperationalMetricsSnapshot(
            uptime_seconds=uptime,
            total_requests=total,
            successful_predictions=success,
            failed_requests=failed,
            prediction_classes=pred_classes,
            latency_ms=lat_summary,
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    def to_prometheus_format(self) -> str:
        """Formats metrics in Prometheus plaintext exposition standard."""
        snap = self.get_metrics_snapshot()
        lines = [
            "# HELP mlops_inference_requests_total Total number of inference requests",
            "# TYPE mlops_inference_requests_total counter",
            f"mlops_inference_requests_total {snap.total_requests}",
            "# HELP mlops_inference_predictions_success_total Successful predictions",
            "# TYPE mlops_inference_predictions_success_total counter",
            f"mlops_inference_predictions_success_total {snap.successful_predictions}",
            "# HELP mlops_inference_requests_failed_total Failed requests",
            "# TYPE mlops_inference_requests_failed_total counter",
            f"mlops_inference_requests_failed_total {snap.failed_requests}",
            "# HELP mlops_inference_latency_p50_ms 50th percentile latency in ms",
            "# TYPE mlops_inference_latency_p50_ms gauge",
            f"mlops_inference_latency_p50_ms {snap.latency_ms['p50_ms']}",
            "# HELP mlops_inference_latency_p95_ms 95th percentile latency in ms",
            "# TYPE mlops_inference_latency_p95_ms gauge",
            f"mlops_inference_latency_p95_ms {snap.latency_ms['p95_ms']}",
            "# HELP mlops_prediction_class_count Breakdown of predictions by predicted class",
            "# TYPE mlops_prediction_class_count counter",
            f'mlops_prediction_class_count{{class="0_retained"}} {snap.prediction_classes.get("0", 0)}',
            f'mlops_prediction_class_count{{class="1_churn"}} {snap.prediction_classes.get("1", 0)}',
            "# HELP mlops_service_uptime_seconds Uptime in seconds",
            "# TYPE mlops_service_uptime_seconds gauge",
            f"mlops_service_uptime_seconds {snap.uptime_seconds}",
        ]
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Resets tracking state (primarily for automated testing)."""
        with self._lock:
            self.start_time = time.time()
            self.total_requests = 0
            self.successful_predictions = 0
            self.failed_requests = 0
            self.prediction_classes = {"0": 0, "1": 0}
            self.latency_samples.clear()


# Global singleton instance for the serving runtime
metrics_tracker = MetricsTracker()
