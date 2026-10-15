"""Unit tests for dataset schema verification, missing columns, and data leakage audit."""

import pandas as pd
import pytest
from mlops_factory.config import DEFAULT_CONFIG, PipelineConfig
from mlops_factory.data.validator import validate_dataset


def test_validate_dataset_on_valid_data(sample_synthetic_dataframe):
    """Validates that a correctly formatted dataframe passes validation."""
    result = validate_dataset(sample_synthetic_dataframe, config=DEFAULT_CONFIG)
    assert result.is_valid is True
    assert result.row_count == len(sample_synthetic_dataframe)
    assert len(result.missing_required_columns) == 0
    assert "Yes" in result.target_distribution
    assert "No" in result.target_distribution


def test_validate_dataset_missing_target(sample_synthetic_dataframe):
    """Ensures validation fails when target column is absent."""
    df_no_target = sample_synthetic_dataframe.drop(columns=["Churn Label"])
    result = validate_dataset(df_no_target, config=DEFAULT_CONFIG)

    assert result.is_valid is False
    assert any("Target column 'Churn Label' is missing" in err for err in result.validation_errors)


def test_validate_dataset_missing_required_features(sample_synthetic_dataframe):
    """Ensures validation fails when required features are missing."""
    df_missing_features = sample_synthetic_dataframe.drop(columns=["Monthly Charge", "Contract"])
    result = validate_dataset(df_missing_features, config=DEFAULT_CONFIG)

    assert result.is_valid is False
    assert "Monthly Charge" in result.missing_required_columns
    assert "Contract" in result.missing_required_columns


def test_validate_dataset_audits_leakage_columns(sample_synthetic_dataframe):
    """Verifies that all known leakage columns are detected and flagged in validation report."""
    result = validate_dataset(sample_synthetic_dataframe, config=DEFAULT_CONFIG)

    # Synthetic dataframe contains: Customer ID, Country, State, Quarter, Customer Status, Churn Reason, Churn Category, etc.
    assert len(result.identified_leakage_columns) > 0
    assert "Customer ID" in result.identified_leakage_columns
    assert "Customer Status" in result.identified_leakage_columns
    assert "Churn Reason" in result.identified_leakage_columns
    assert "Satisfaction Score" in result.identified_leakage_columns
    assert any("known target/identifier leakage columns" in warn for warn in result.validation_warnings)


def test_validate_dataset_detects_duplicates(sample_synthetic_dataframe):
    """Verifies duplicate row detection."""
    df_with_dups = pd.concat([sample_synthetic_dataframe, sample_synthetic_dataframe.iloc[:5]], ignore_index=True)
    result = validate_dataset(df_with_dups, config=DEFAULT_CONFIG)

    assert result.duplicate_rows_count == 5
    assert any("duplicate rows" in warn for warn in result.validation_warnings)
