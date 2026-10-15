"""Data quality validation, schema enforcement, and leakage audit."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List
import pandas as pd
from mlops_factory.config import DEFAULT_CONFIG, PipelineConfig


@dataclass
class ColumnQualitySummary:
    """Quality metrics for an individual column."""
    name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: List[Any]


@dataclass
class DataValidationResult:
    """Consolidated dataset validation outcome."""
    is_valid: bool
    row_count: int
    column_count: int
    missing_required_columns: List[str]
    identified_leakage_columns: List[str]
    duplicate_rows_count: int
    target_distribution: Dict[str, int]
    target_proportions: Dict[str, float]
    column_summaries: Dict[str, ColumnQualitySummary]
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def validate_dataset(
    df: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
    strict_leakage_rejection: bool = False,
) -> DataValidationResult:
    """
    Validates the dataset against required features, checks target column integrity,
    audits potential target/identifier leakage, and flags data-quality anomalies.
    """
    validation_errors: List[str] = []
    validation_warnings: List[str] = []

    # 1. Target column presence and distribution
    if config.target_column not in df.columns:
        validation_errors.append(
            f"Target column '{config.target_column}' is missing from the dataset. "
            f"Expected binary target representing customer churn."
        )
        target_dist = {}
        target_prop = {}
    else:
        target_series = df[config.target_column].dropna()
        target_dist = target_series.value_counts().to_dict()
        total_valid = len(target_series)
        target_prop = {str(k): round(v / total_valid, 4) for k, v in target_dist.items()} if total_valid > 0 else {}

        # Check binary expectation
        unique_targets = set(target_series.unique())
        expected_targets = {config.positive_target_value, config.negative_target_value}
        if not expected_targets.issubset(unique_targets):
            validation_warnings.append(
                f"Target column contains values {unique_targets}, expected {expected_targets}."
            )

    # 2. Required feature columns check
    all_required_features = set(config.numeric_features + config.categorical_features)
    missing_required = [col for col in all_required_features if col not in df.columns]
    if missing_required:
        validation_errors.append(
            f"Dataset is missing {len(missing_required)} required feature columns: {missing_required}"
        )

    # 3. Leakage audit
    identified_leakage = [col for col in config.leakage_columns if col in df.columns and col != config.target_column]
    if identified_leakage:
        msg = (
            f"Dataset contains {len(identified_leakage)} known target/identifier leakage columns: "
            f"{identified_leakage}. These MUST be dropped prior to model feature engineering."
        )
        if strict_leakage_rejection:
            validation_errors.append(msg)
        else:
            validation_warnings.append(msg)

    # 4. Duplicate rows
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        validation_warnings.append(f"Dataset contains {duplicate_rows} duplicate rows.")

    # 5. Column-wise statistics
    col_summaries: Dict[str, ColumnQualitySummary] = {}
    total_rows = len(df)
    for col in df.columns:
        null_cnt = int(df[col].isnull().sum())
        null_pct = round((null_cnt / total_rows) * 100, 2) if total_rows > 0 else 0.0
        n_unique = int(df[col].nunique())
        samples = [str(x) for x in df[col].dropna().unique()[:3].tolist()]

        col_summaries[col] = ColumnQualitySummary(
            name=col,
            dtype=str(df[col].dtype),
            null_count=null_cnt,
            null_percentage=null_pct,
            unique_count=n_unique,
            sample_values=samples,
        )

    is_valid = len(validation_errors) == 0

    return DataValidationResult(
        is_valid=is_valid,
        row_count=total_rows,
        column_count=len(df.columns),
        missing_required_columns=missing_required,
        identified_leakage_columns=identified_leakage,
        duplicate_rows_count=duplicate_rows,
        target_distribution=target_dist,
        target_proportions=target_prop,
        column_summaries=col_summaries,
        validation_errors=validation_errors,
        validation_warnings=validation_warnings,
    )
