"""Configuration settings and schema contracts for MLOps Factory."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class QualityGateConfig:
    """Acceptance criteria thresholds for candidate models."""
    min_roc_auc: float = 0.82
    min_recall: float = 0.70
    min_precision: float = 0.50
    min_f1_score: float = 0.60
    min_accuracy: float = 0.75
    max_training_duration_seconds: float = 120.0


@dataclass(frozen=True)
class DriftConfig:
    """Thresholds for operational drift detection."""
    psi_warning_threshold: float = 0.10
    psi_critical_threshold: float = 0.25
    ks_alpha: float = 0.05


@dataclass(frozen=True)
class PipelineConfig:
    """Master configuration for the MLOps training, validation, and serving pipeline."""
    # Data paths
    raw_data_filename: str = "telco.csv"
    data_dir: Path = Path(".")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    docs_dir: Path = Path("docs")

    # Target definition
    target_column: str = "Churn Label"
    positive_target_value: str = "Yes"
    negative_target_value: str = "No"

    # Leakage and zero-variance columns to strictly exclude from model features
    leakage_columns: List[str] = field(default_factory=lambda: [
        "Customer ID",                # Unique identifier / primary key leakage
        "Country",                    # Constant ('United States') - zero variance
        "State",                      # Constant ('California') - zero variance
        "Quarter",                    # Constant ('Q3') - zero variance
        "City",                       # High-cardinality geographic identifier (1,106 unique)
        "Zip Code",                   # High-cardinality geographic identifier (1,626 unique)
        "Latitude",                   # Geospatial coordinate identifier
        "Longitude",                  # Geospatial coordinate identifier
        "Customer Status",            # Target duplicate ('Stayed', 'Churned', 'Joined')
        "Churn Label",                # Direct ground-truth target
        "Churn Score",                # Pre-existing risk score computed by vendor (target leakage)
        "Churn Category",             # Post-event reason category (only non-null on churned customers)
        "Churn Reason",               # Post-event exit interview notes (only non-null on churned customers)
        "Satisfaction Score",         # Post-event exit survey rating (1-5); strictly separates churners
    ])

    # 14 Numeric features
    numeric_features: List[str] = field(default_factory=lambda: [
        "Age",
        "Number of Dependents",
        "Population",
        "Number of Referrals",
        "Tenure in Months",
        "Avg Monthly Long Distance Charges",
        "Avg Monthly GB Download",
        "Monthly Charge",
        "Total Charges",
        "Total Refunds",
        "Total Extra Data Charges",
        "Total Long Distance Charges",
        "Total Revenue",
        "CLTV",
    ])

    # 22 Categorical features
    categorical_features: List[str] = field(default_factory=lambda: [
        "Gender",
        "Under 30",
        "Senior Citizen",
        "Married",
        "Dependents",
        "Referred a Friend",
        "Offer",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Internet Type",
        "Online Security",
        "Online Backup",
        "Device Protection Plan",
        "Premium Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Streaming Music",
        "Unlimited Data",
        "Contract",
        "Paperless Billing",
        "Payment Method",
    ])

    # Train / Val / Test splitting
    test_size: float = 0.20
    val_size: float = 0.15
    random_seed: int = 42

    # Quality Gate and Drift thresholds
    quality_gates: QualityGateConfig = field(default_factory=QualityGateConfig)
    drift_config: DriftConfig = field(default_factory=DriftConfig)

    # API configuration
    api_title: str = "MLOps Factory — Telco Churn Prediction API"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


DEFAULT_CONFIG = PipelineConfig()
