"""Unit tests for operational metrics telemetry and statistical distribution drift detection."""

import numpy as np
import pandas as pd
from mlops_factory.features.preprocessor import clean_inference_features
from mlops_factory.monitoring.drift import (
    calculate_categorical_psi,
    calculate_psi,
    detect_data_drift,
)
from mlops_factory.monitoring.tracker import MetricsTracker


def test_calculate_psi_identical_distributions():
    """Validates that PSI is approximately zero for identical distributions."""
    rng = np.random.RandomState(42)
    data = rng.normal(100, 15, size=1000)

    psi_val = calculate_psi(data, data)
    assert 0.0 <= psi_val < 0.05


def test_calculate_psi_shifted_distributions():
    """Validates that a significant distribution shift yields elevated PSI (> 0.25)."""
    rng = np.random.RandomState(42)
    base_data = rng.normal(100, 10, size=1000)
    shifted_data = rng.normal(150, 25, size=1000)

    psi_val = calculate_psi(base_data, shifted_data)
    assert psi_val > 0.25


def test_calculate_categorical_psi():
    """Tests PSI computation on discrete categories."""
    base = pd.Series(["A", "A", "B", "C", "A"] * 200)
    shifted = pd.Series(["C", "C", "B", "C", "C"] * 200)

    psi_val = calculate_categorical_psi(base, shifted)
    assert psi_val > 0.10


def test_detect_data_drift_report(sample_synthetic_dataframe):
    """Tests comprehensive drift report generation and alerting."""
    ref_df = clean_inference_features(sample_synthetic_dataframe.iloc[:30])
    curr_df = clean_inference_features(sample_synthetic_dataframe.iloc[30:])

    report = detect_data_drift(curr_df, ref_df)
    assert report.sample_size == 30
    assert report.reference_sample_size == 30
    assert report.features_analyzed_count > 0
    assert isinstance(report.overall_drift_detected, bool)
    assert "disclaimer" in report.to_dict()
    assert "not available at inference time" in report.disclaimer


def test_metrics_tracker_telemetry():
    """Tests MetricsTracker recording, percentiles, and reset capabilities."""
    tracker = MetricsTracker()
    tracker.reset()

    tracker.record_prediction(prediction_class=0, latency_seconds=0.010)
    tracker.record_prediction(prediction_class=1, latency_seconds=0.020)
    tracker.record_prediction(prediction_class=1, latency_seconds=0.030)
    tracker.record_error()

    snap = tracker.get_metrics_snapshot()
    assert snap.total_requests == 4
    assert snap.successful_predictions == 3
    assert snap.failed_requests == 1
    assert snap.prediction_classes["0"] == 1
    assert snap.prediction_classes["1"] == 2
    assert snap.latency_ms["min_ms"] >= 10.0
    assert snap.latency_ms["max_ms"] <= 35.0

    prom_text = tracker.to_prometheus_format()
    assert "mlops_inference_requests_total 4" in prom_text
    assert "mlops_inference_requests_failed_total 1" in prom_text

    tracker.reset()
    snap_after = tracker.get_metrics_snapshot()
    assert snap_after.total_requests == 0
