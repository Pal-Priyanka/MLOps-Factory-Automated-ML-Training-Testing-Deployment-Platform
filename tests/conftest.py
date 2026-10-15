"""Shared pytest test fixtures, mock data generators, and test environments."""

import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import pytest

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mlops_factory.config import DEFAULT_CONFIG, PipelineConfig
from mlops_factory.features.preprocessor import build_preprocessor_pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def sample_valid_customer_payload() -> Dict[str, Any]:
    """Provides a realistic single-customer inference request payload."""
    return {
        "Gender": "Female",
        "Age": 55,
        "Under 30": "No",
        "Senior Citizen": "No",
        "Married": "Yes",
        "Dependents": "No",
        "Number of Dependents": 0,
        "Population": 32000,
        "Referred a Friend": "No",
        "Number of Referrals": 0,
        "Tenure in Months": 3,
        "Offer": "Offer E",
        "Phone Service": "Yes",
        "Avg Monthly Long Distance Charges": 24.5,
        "Multiple Lines": "No",
        "Internet Service": "Yes",
        "Internet Type": "Fiber Optic",
        "Avg Monthly GB Download": 18,
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection Plan": "No",
        "Premium Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Streaming Music": "No",
        "Unlimited Data": "Yes",
        "Contract": "Month-to-Month",
        "Paperless Billing": "Yes",
        "Payment Method": "Bank Withdrawal",
        "Monthly Charge": 85.50,
        "Total Charges": 256.50,
        "Total Refunds": 0.0,
        "Total Extra Data Charges": 0,
        "Total Long Distance Charges": 73.5,
        "Total Revenue": 330.0,
        "CLTV": 3800,
    }


@pytest.fixture
def sample_synthetic_dataframe() -> pd.DataFrame:
    """Generates a small deterministic dataframe covering all required features and targets."""
    rng = np.random.RandomState(42)
    n = 60

    data = {
        # Leakage / Identifiers (to test stripping)
        "Customer ID": [f"TEST-{i:04d}" for i in range(n)],
        "Country": ["United States"] * n,
        "State": ["California"] * n,
        "Quarter": ["Q3"] * n,
        "City": ["Testville"] * n,
        "Zip Code": [90001 + i for i in range(n)],
        "Latitude": [34.0 + i * 0.01 for i in range(n)],
        "Longitude": [-118.0 - i * 0.01 for i in range(n)],
        "Customer Status": rng.choice(["Stayed", "Churned"], size=n),
        "Churn Category": [None if i % 2 == 0 else "Competitor" for i in range(n)],
        "Churn Reason": [None if i % 2 == 0 else "Price too high" for i in range(n)],
        "Churn Score": rng.randint(10, 95, size=n),
        "Satisfaction Score": rng.choice([1, 2, 3, 4, 5], size=n),
        # Target
        "Churn Label": rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7]),
        # Numeric Features
        "Age": rng.randint(18, 80, size=n),
        "Number of Dependents": rng.randint(0, 5, size=n),
        "Population": rng.randint(5000, 80000, size=n),
        "Number of Referrals": rng.randint(0, 10, size=n),
        "Tenure in Months": rng.randint(1, 72, size=n),
        "Avg Monthly Long Distance Charges": rng.uniform(0.0, 50.0, size=n),
        "Avg Monthly GB Download": rng.randint(0, 80, size=n),
        "Monthly Charge": rng.uniform(20.0, 110.0, size=n),
        "Total Charges": rng.uniform(20.0, 5000.0, size=n),
        "Total Refunds": rng.uniform(0.0, 50.0, size=n),
        "Total Extra Data Charges": rng.choice([0, 10, 20], size=n),
        "Total Long Distance Charges": rng.uniform(0.0, 1000.0, size=n),
        "Total Revenue": rng.uniform(50.0, 6000.0, size=n),
        "CLTV": rng.randint(2000, 6000, size=n),
        # Categorical Features
        "Gender": rng.choice(["Male", "Female"], size=n),
        "Under 30": rng.choice(["Yes", "No"], size=n),
        "Senior Citizen": rng.choice(["Yes", "No"], size=n),
        "Married": rng.choice(["Yes", "No"], size=n),
        "Dependents": rng.choice(["Yes", "No"], size=n),
        "Referred a Friend": rng.choice(["Yes", "No"], size=n),
        "Offer": rng.choice(["Offer A", "Offer B", "None"], size=n),
        "Phone Service": rng.choice(["Yes", "No"], size=n),
        "Multiple Lines": rng.choice(["Yes", "No"], size=n),
        "Internet Service": rng.choice(["Yes", "No"], size=n),
        "Internet Type": rng.choice(["DSL", "Fiber Optic", "None"], size=n),
        "Online Security": rng.choice(["Yes", "No"], size=n),
        "Online Backup": rng.choice(["Yes", "No"], size=n),
        "Device Protection Plan": rng.choice(["Yes", "No"], size=n),
        "Premium Tech Support": rng.choice(["Yes", "No"], size=n),
        "Streaming TV": rng.choice(["Yes", "No"], size=n),
        "Streaming Movies": rng.choice(["Yes", "No"], size=n),
        "Streaming Music": rng.choice(["Yes", "No"], size=n),
        "Unlimited Data": rng.choice(["Yes", "No"], size=n),
        "Contract": rng.choice(["Month-to-Month", "One Year", "Two Year"], size=n),
        "Paperless Billing": rng.choice(["Yes", "No"], size=n),
        "Payment Method": rng.choice(["Bank Withdrawal", "Credit Card", "Mailed Check"], size=n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def fitted_mock_pipeline(sample_synthetic_dataframe) -> Pipeline:
    """Provides a quickly fitted pipeline on synthetic data for unit tests."""
    from mlops_factory.features.preprocessor import clean_inference_features

    df = sample_synthetic_dataframe
    X = clean_inference_features(df, config=DEFAULT_CONFIG)
    y = (df["Churn Label"] == "Yes").astype(int)

    preprocessor = build_preprocessor_pipeline(
        DEFAULT_CONFIG.numeric_features,
        DEFAULT_CONFIG.categorical_features,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=100, random_state=42)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline
