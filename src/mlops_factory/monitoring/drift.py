"""Input data distribution check and statistical drift detection."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from mlops_factory.config import DEFAULT_CONFIG, DriftConfig


@dataclass
class FeatureDriftResult:
    feature_name: str
    feature_type: str  # 'numeric' or 'categorical'
    psi_score: float
    ks_statistic: Optional[float]
    ks_p_value: Optional[float]
    drift_level: str  # 'NONE', 'MODERATE', 'CRITICAL'
    alert: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Comprehensive statistical drift analysis report."""
    timestamp_utc: str
    sample_size: int
    reference_sample_size: int
    features_analyzed_count: int
    features_drifting_count: int
    overall_drift_detected: bool
    feature_results: List[FeatureDriftResult]
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "sample_size": self.sample_size,
            "reference_sample_size": self.reference_sample_size,
            "features_analyzed_count": self.features_analyzed_count,
            "features_drifting_count": self.features_drifting_count,
            "overall_drift_detected": self.overall_drift_detected,
            "feature_results": [asdict(r) for r in self.feature_results],
            "disclaimer": self.disclaimer,
        }


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Calculates Population Stability Index (PSI) between reference and production sample.

    PSI interpretation:
      < 0.10: No significant distribution change.
      0.10 - 0.25: Moderate shift; investigate.
      >= 0.25: Significant drift; model retraining recommended.
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins on expected distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    bins = np.unique(bins)

    if len(bins) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    # Convert to fractions with smoothing epsilon
    expected_frac = expected_counts / len(expected)
    actual_frac = actual_counts / len(actual)

    expected_frac = np.where(expected_frac == 0, epsilon, expected_frac)
    actual_frac = np.where(actual_frac == 0, epsilon, actual_frac)

    psi_val = np.sum((actual_frac - expected_frac) * np.log(actual_frac / expected_frac))
    return float(max(0.0, psi_val))


def calculate_categorical_psi(
    expected_series: pd.Series,
    actual_series: pd.Series,
    epsilon: float = 1e-4,
) -> float:
    """Calculates PSI for discrete/categorical features across shared categories."""
    exp_counts = expected_series.value_counts(normalize=True)
    act_counts = actual_series.value_counts(normalize=True)

    all_categories = sorted(list(set(exp_counts.index).union(set(act_counts.index))))
    if not all_categories:
        return 0.0

    exp_fracs = np.array([exp_counts.get(cat, epsilon) for cat in all_categories])
    act_fracs = np.array([act_counts.get(cat, epsilon) for cat in all_categories])

    psi_val = np.sum((act_fracs - exp_fracs) * np.log(act_fracs / exp_fracs))
    return float(max(0.0, psi_val))


def detect_data_drift(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    config: DriftConfig = DEFAULT_CONFIG.drift_config,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> DriftReport:
    """
    Performs dual-method distribution drift detection (PSI + two-sample KS-test)
    comparing incoming inference features against training baseline reference.
    """
    if numeric_features is None:
        numeric_features = DEFAULT_CONFIG.numeric_features
    if categorical_features is None:
        categorical_features = DEFAULT_CONFIG.categorical_features

    feature_results: List[FeatureDriftResult] = []
    drifting_count = 0

    # 1. Numeric features evaluation
    for col in numeric_features:
        if col not in current_df.columns or col not in reference_df.columns:
            continue

        ref_vals = pd.to_numeric(reference_df[col], errors="coerce").dropna().values
        curr_vals = pd.to_numeric(current_df[col], errors="coerce").dropna().values

        if len(ref_vals) < 5 or len(curr_vals) < 5:
            continue

        psi = round(calculate_psi(ref_vals, curr_vals), 4)

        # KS test
        ks_stat, ks_p = ks_2samp(ref_vals, curr_vals)
        ks_stat = round(float(ks_stat), 4)
        ks_p = round(float(ks_p), 4)

        if psi >= config.psi_critical_threshold:
            level = "CRITICAL"
            alert = True
        elif psi >= config.psi_warning_threshold:
            level = "MODERATE"
            alert = True
        else:
            level = "NONE"
            alert = False

        if alert:
            drifting_count += 1

        feature_results.append(
            FeatureDriftResult(
                feature_name=col,
                feature_type="numeric",
                psi_score=psi,
                ks_statistic=ks_stat,
                ks_p_value=ks_p,
                drift_level=level,
                alert=alert,
                details={
                    "ref_mean": round(float(np.mean(ref_vals)), 2),
                    "curr_mean": round(float(np.mean(curr_vals)), 2),
                },
            )
        )

    # 2. Categorical features evaluation
    for col in categorical_features:
        if col not in current_df.columns or col not in reference_df.columns:
            continue

        ref_series = reference_df[col].astype(str)
        curr_series = current_df[col].astype(str)

        psi = round(calculate_categorical_psi(ref_series, curr_series), 4)

        if psi >= config.psi_critical_threshold:
            level = "CRITICAL"
            alert = True
        elif psi >= config.psi_warning_threshold:
            level = "MODERATE"
            alert = True
        else:
            level = "NONE"
            alert = False

        if alert:
            drifting_count += 1

        feature_results.append(
            FeatureDriftResult(
                feature_name=col,
                feature_type="categorical",
                psi_score=psi,
                ks_statistic=None,
                ks_p_value=None,
                drift_level=level,
                alert=alert,
                details={"categories_analyzed": len(set(ref_series).union(set(curr_series)))},
            )
        )

    overall_drift = drifting_count > 0

    disclaimer = (
        "Statistical data drift signals indicate input distribution shifts compared to training reference. "
        "It does NOT represent verified model accuracy degradation, as ground-truth churn labels are not available "
        "at inference time. Re-evaluating or retraining should be scheduled if critical drift persists."
    )

    return DriftReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        sample_size=len(current_df),
        reference_sample_size=len(reference_df),
        features_analyzed_count=len(feature_results),
        features_drifting_count=drifting_count,
        overall_drift_detected=overall_drift,
        feature_results=feature_results,
        disclaimer=disclaimer,
    )
