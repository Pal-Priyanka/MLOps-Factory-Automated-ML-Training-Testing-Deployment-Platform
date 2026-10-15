"""Leakage-safe scikit-learn preprocessing pipeline."""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_factory.config import DEFAULT_CONFIG, PipelineConfig


def build_preprocessor_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that imputes and scales numeric features,
    and imputes and one-hot encodes categorical features.

    Safe against unknown categories in production via `handle_unknown='ignore'`.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # Drop any unconfigured or leakage columns safely
    )

    return preprocessor


def clean_inference_features(
    df: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Ensures input inference data matches expected feature schema,
    sanitizes known categorical nulls, and eliminates any accidentally passed leakage columns.
    """
    df_clean = df.copy()

    # Drop any leakage or identifier columns if accidentally provided
    drop_candidates = [col for col in config.leakage_columns if col in df_clean.columns]
    if drop_candidates:
        df_clean = df_clean.drop(columns=drop_candidates)

    # Fill domain-specific known optional categories
    for col in config.categorical_features:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("None").astype(str)

    # Cast numeric columns appropriately
    for col in config.numeric_features:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Retain strictly the union of configured numeric and categorical features
    expected_cols = config.numeric_features + config.categorical_features
    # If any expected feature is missing in inference, add it as NaN for imputer to handle
    for col in expected_cols:
        if col not in df_clean.columns:
            df_clean[col] = np.nan

    return df_clean[expected_cols]


def prepare_features_and_target(
    df: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts strictly leak-free feature matrix X and binary target vector y.
    Guarantees no target or identifier leakage is passed to model training.
    """
    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' is missing from dataframe.")

    # Ground truth binary target (1 = Churned, 0 = Retained)
    y = (df[config.target_column].astype(str).str.strip().str.lower() == config.positive_target_value.lower()).astype(int)

    # Isolate strictly allowed feature columns
    X = clean_inference_features(df, config=config)

    return X, y
