"""Unit tests for feature preprocessing, pipeline fitting, and leakage elimination."""

import numpy as np
import pandas as pd
import pytest
from mlops_factory.config import DEFAULT_CONFIG
from mlops_factory.features.preprocessor import (
    build_preprocessor_pipeline,
    clean_inference_features,
    prepare_features_and_target,
)


def test_clean_inference_features_drops_leakage(sample_synthetic_dataframe):
    """Ensures clean_inference_features strictly eliminates leakage columns."""
    df_clean = clean_inference_features(sample_synthetic_dataframe, config=DEFAULT_CONFIG)

    for leakage_col in DEFAULT_CONFIG.leakage_columns:
        assert leakage_col not in df_clean.columns

    # Verify only required features are present
    expected_cols = DEFAULT_CONFIG.numeric_features + DEFAULT_CONFIG.categorical_features
    assert set(df_clean.columns) == set(expected_cols)


def test_preprocessor_pipeline_transform_shape(sample_synthetic_dataframe):
    """Tests fitting and transforming via scikit-learn ColumnTransformer."""
    X, y = prepare_features_and_target(sample_synthetic_dataframe, config=DEFAULT_CONFIG)
    preprocessor = build_preprocessor_pipeline(
        DEFAULT_CONFIG.numeric_features,
        DEFAULT_CONFIG.categorical_features,
    )

    X_transformed = preprocessor.fit_transform(X)
    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == len(sample_synthetic_dataframe)
    # Number of columns should exceed number of raw features due to one-hot encoding
    assert X_transformed.shape[1] > len(X.columns)


def test_preprocessor_handles_unseen_categories(sample_synthetic_dataframe):
    """Validates handle_unknown='ignore' behavior on novel production categories."""
    X, y = prepare_features_and_target(sample_synthetic_dataframe, config=DEFAULT_CONFIG)
    preprocessor = build_preprocessor_pipeline(
        DEFAULT_CONFIG.numeric_features,
        DEFAULT_CONFIG.categorical_features,
    )
    preprocessor.fit(X)

    # Introduce novel category
    X_novel = X.copy()
    X_novel.loc[0, "Payment Method"] = "Cryptocurrency Wallet (Novel)"
    X_novel.loc[0, "Contract"] = "Centennial 100-Year Plan (Novel)"

    transformed_novel = preprocessor.transform(X_novel)
    assert transformed_novel.shape[0] == len(X_novel)
    assert not np.isnan(transformed_novel).any()


def test_prepare_features_and_target_binary_mapping(sample_synthetic_dataframe):
    """Verifies that ground truth target is properly mapped to 0 and 1."""
    X, y = prepare_features_and_target(sample_synthetic_dataframe, config=DEFAULT_CONFIG)
    assert set(y.unique()).issubset({0, 1})
    assert len(y) == len(sample_synthetic_dataframe)
