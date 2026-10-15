"""Simulates production inference traffic to demonstrate statistical drift detection."""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mlops_factory.config import DEFAULT_CONFIG
from mlops_factory.data.discovery import discover_dataset
from mlops_factory.features.preprocessor import clean_inference_features
from mlops_factory.monitoring.drift import detect_data_drift


def run_simulation():
    dataset_path = discover_dataset()
    df = pd.read_csv(dataset_path)

    # Reference data: first 1000 rows
    reference_sample = clean_inference_features(df.iloc[:1500], config=DEFAULT_CONFIG)

    # Production scenario A: Normal Traffic (Rows 1500 to 2500)
    normal_sample = clean_inference_features(df.iloc[1500:2500], config=DEFAULT_CONFIG)
    normal_report = detect_data_drift(normal_sample, reference_sample)

    print("\n" + "=" * 60)
    print("SCENARIO A: NORMAL INFERENCE TRAFFIC")
    print("=" * 60)
    print(f"Sample size: {normal_report.sample_size}")
    print(f"Features drifting: {normal_report.features_drifting_count}/{normal_report.features_analyzed_count}")
    print(f"Overall Drift Alert: {'TRIGGERED' if normal_report.overall_drift_detected else 'CLEAR'}")

    # Production scenario B: Artificially Drifted Traffic
    # E.g. Macroeconomic shift: shorter tenures, higher monthly charges, shifted contracts
    drifted_df = df.iloc[2500:3500].copy()
    drifted_df["Monthly Charge"] = drifted_df["Monthly Charge"] * 1.6 + 25.0
    drifted_df["Tenure in Months"] = np.maximum(1, (drifted_df["Tenure in Months"] * 0.3).astype(int))
    drifted_df["Contract"] = "Month-to-Month"  # Severe shift to month-to-month contracts

    drifted_sample = clean_inference_features(drifted_df, config=DEFAULT_CONFIG)
    drift_report = detect_data_drift(drifted_sample, reference_sample)

    print("\n" + "=" * 60)
    print("SCENARIO B: SIMULATED COHORT DRIFT (Economic/Pricing Shift)")
    print("=" * 60)
    print(f"Sample size: {drift_report.sample_size}")
    print(f"Features drifting: {drift_report.features_drifting_count}/{drift_report.features_analyzed_count}")
    print(f"Overall Drift Alert: {'TRIGGERED (WARNING/CRITICAL)' if drift_report.overall_drift_detected else 'CLEAR'}")
    print("\nDrifting Features Detail:")
    for res in drift_report.feature_results:
        if res.alert:
            ks_info = f", KS p-value: {res.ks_p_value}" if res.ks_p_value is not None else ""
            print(f"  - [{res.drift_level}] Feature '{res.feature_name}': PSI = {res.psi_score:.4f}{ks_info}")

    # Save drift report artifact
    reports_dir = DEFAULT_CONFIG.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "drift_report.json"
    with open(report_file, "w") as f:
        json.dump(drift_report.to_dict(), f, indent=2)

    print(f"\n[*] Full drift report saved to: {report_file.resolve()}")
    print(f"[*] Note: {drift_report.disclaimer}")


if __name__ == "__main__":
    run_simulation()
