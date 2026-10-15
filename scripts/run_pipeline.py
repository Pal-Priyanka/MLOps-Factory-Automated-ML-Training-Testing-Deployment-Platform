"""Command-line script to run full training, evaluation, quality gating, and reporting."""

import argparse
import json
import sys
from pathlib import Path

# Add src to python path for direct script invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mlops_factory.config import DEFAULT_CONFIG
from mlops_factory.data.discovery import discover_dataset
from mlops_factory.training.train import TrainingOrchestrator


def generate_markdown_reports(results: dict, reports_dir: Path):
    """Generates detailed markdown evaluation and data quality reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    val_res = results["validation_result"]
    base_test = results["baseline_test_metrics"]
    cand_test = results["candidate_test_metrics"]
    selected_name = results["selected_model_name"]
    gates = results["quality_gates"]

    # 1. Model Evaluation Report
    eval_report_path = reports_dir / "model_evaluation_report.md"
    winner_test = cand_test if selected_name == "Candidate_HistGradientBoosting" else base_test

    with open(eval_report_path, "w", encoding="utf-8") as f:
        f.write("# Model Evaluation & Benchmarking Report\n\n")
        f.write(f"**Selected Production Model**: `{selected_name}`  \n")
        f.write(f"**Quality Gate Status**: {'PASSED (Production-Ready)' if gates.passed else 'FAILED (Blocked from Production)'}  \n\n")

        f.write("## 1. Executive Summary & Comparison\n\n")
        f.write("| Metric | Baseline (Logistic Regression) | Candidate (HistGradientBoosting) | Delta (Candidate vs Baseline) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **ROC-AUC** | {base_test.roc_auc:.4f} | {cand_test.roc_auc:.4f} | {cand_test.roc_auc - base_test.roc_auc:+.4f} |\n")
        f.write(f"| **PR-AUC** | {base_test.pr_auc:.4f} | {cand_test.pr_auc:.4f} | {cand_test.pr_auc - base_test.pr_auc:+.4f} |\n")
        f.write(f"| **Accuracy** | {base_test.accuracy:.4f} | {cand_test.accuracy:.4f} | {cand_test.accuracy - base_test.accuracy:+.4f} |\n")
        f.write(f"| **Precision (Churn)** | {base_test.precision:.4f} | {cand_test.precision:.4f} | {cand_test.precision - base_test.precision:+.4f} |\n")
        f.write(f"| **Recall (Churn)** | {base_test.recall:.4f} | {cand_test.recall:.4f} | {cand_test.recall - base_test.recall:+.4f} |\n")
        f.write(f"| **F1-Score** | {base_test.f1_score:.4f} | {cand_test.f1_score:.4f} | {cand_test.f1_score - base_test.f1_score:+.4f} |\n")
        f.write(f"| **Training Duration** | {base_test.training_duration_seconds:.2f}s | {cand_test.training_duration_seconds:.2f}s | {cand_test.training_duration_seconds - base_test.training_duration_seconds:+.2f}s |\n\n")

        f.write("## 2. Confusion Matrix on Holdout Test Set\n\n")
        cm = winner_test.confusion_matrix
        f.write(f"Total Test Set Samples: {cm.true_negatives + cm.false_positives + cm.false_negatives + cm.true_positives}\n\n")
        f.write("| Actual / Predicted | Predicted: Retained (0) | Predicted: Churned (1) |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **Actual: Retained (0)** | True Negatives: `{cm.true_negatives}` | False Positives (False Alarms): `{cm.false_positives}` |\n")
        f.write(f"| **Actual: Churned (1)** | False Negatives (Missed Churners): `{cm.false_negatives}` | True Positives (Caught Churners): `{cm.true_positives}` |\n\n")

        f.write("## 3. Business Trade-Off Analysis\n\n")
        f.write(f"- {winner_test.business_tradeoff_summary.get('interpretation', '')}\n")
        f.write("- **False Negative Impact**: A missed churner leaves the platform, losing recurring monthly revenue and customer lifetime value (CLTV).\n")
        f.write("- **False Positive Impact**: A retained customer receives a proactive retention offer (e.g. discounted rate or promotional bundle), incurring minor marketing expense.\n")
        f.write("- **Strategy**: Because customer acquisition cost (CAC) significantly exceeds retention promotion cost, the pipeline optimizes for high Recall while maintaining an acceptable Precision threshold (>= 0.50).\n\n")

        f.write("## 4. Model Quality Gates Verification\n\n")
        f.write("| Gate Criterion | Target Threshold | Observed Value | Result |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for check in gates.checks:
            status_symbol = "PASSED" if check.passed else "FAILED"
            f.write(f"| {check.metric_name} | {check.comparison_operator} {check.threshold_value} | {check.observed_value} | `{status_symbol}` |\n")

        f.write("\n## 5. Top Feature Importances\n\n")
        f.write("| Rank | Feature | Importance Score |\n")
        f.write("| :---: | :--- | :---: |\n")
        for idx, feat in enumerate(winner_test.top_feature_importances[:10], start=1):
            f.write(f"| {idx} | `{feat['feature']}` | {feat['importance']:.4f} |\n")

    # 2. Data Quality & Leakage Audit Report
    data_report_path = reports_dir / "data_quality_report.md"
    with open(data_report_path, "w", encoding="utf-8") as f:
        f.write("# Data Quality & Leakage Audit Report\n\n")
        f.write(f"**Total Records**: `{val_res.row_count}`  \n")
        f.write(f"**Total Columns**: `{val_res.column_count}`  \n")
        f.write(f"**Duplicate Rows**: `{val_res.duplicate_rows_count}`  \n")
        f.write(f"**Overall Validation Status**: `{'VALID' if val_res.is_valid else 'INVALID'}`  \n\n")

        f.write("## 1. Target Distribution\n\n")
        f.write("| Churn Class | Count | Percentage |\n")
        f.write("| :--- | :---: | :---: |\n")
        for k, v in val_res.target_distribution.items():
            pct = val_res.target_proportions.get(k, 0.0) * 100
            f.write(f"| `{k}` | {v} | {pct:.2f}% |\n")

        f.write("\n## 2. Leakage Audit & Excluded Columns\n\n")
        f.write("The following columns were identified and strictly removed from the model feature matrix to prevent data leakage and artificial inflation of evaluation scores:\n\n")
        f.write("| Excluded Column | Category | Justification |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write("| `Customer ID` | Identifier | Primary key; causes identity memorization rather than generalization. |\n")
        f.write("| `Country`, `State`, `Quarter` | Zero Variance | Constant across all records (`United States`, `California`, `Q3`). |\n")
        f.write("| `City`, `Zip Code`, `Latitude`, `Longitude` | High Cardinality | Spatial identifiers; `Population` provides clean demographic sizing without overfitting. |\n")
        f.write("| `Customer Status` | Target Duplicate | Contains ground-truth statuses: `Stayed`, `Churned`, `Joined`. |\n")
        f.write("| `Churn Category` | Post-Event Leakage | Explains why customer churned; populated exclusively for churned accounts (73.5% missing on retained accounts). |\n")
        f.write("| `Churn Reason` | Post-Event Leakage | Exit interview reason; populated exclusively for churned accounts. |\n")
        f.write("| `Churn Score` | Target Leakage | Vendor-generated predictive risk score (0-100) precalculated by IBM. |\n")
        f.write("| `Satisfaction Score` | Post-Event Leakage | Exit interview survey score (1-5); scores 1 & 2 are 100% churned, scores 4 & 5 are 100% retained. |\n\n")

        f.write("## 3. Configured Features\n\n")
        f.write(f"- **Numeric Features ({len(DEFAULT_CONFIG.numeric_features)})**: `{', '.join(DEFAULT_CONFIG.numeric_features)}`\n")
        f.write(f"- **Categorical Features ({len(DEFAULT_CONFIG.categorical_features)})**: `{', '.join(DEFAULT_CONFIG.categorical_features)}`\n")


def main():
    parser = argparse.ArgumentParser(description="Run MLOps Factory training pipeline")
    parser.add_argument("--data-path", type=str, default=None, help="Explicit path to telco.csv")
    args = parser.parse_args()

    data_path = Path(args.data_path) if args.data_path else discover_dataset()
    print(f"[*] Starting MLOps Factory Pipeline on dataset: {data_path}")

    orchestrator = TrainingOrchestrator(config=DEFAULT_CONFIG)
    results = orchestrator.run(dataset_path=data_path, save_artifacts=True)

    # Generate Markdown reports
    generate_markdown_reports(results, DEFAULT_CONFIG.reports_dir)

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)
    print(f"Dataset SHA-256: {results['dataset_metadata'].sha256_checksum[:16]}...")
    print(f"Selected Model:  {results['selected_model_name']}")
    print(f"Baseline Test ROC-AUC:  {results['baseline_test_metrics'].roc_auc:.4f}")
    print(f"Candidate Test ROC-AUC: {results['candidate_test_metrics'].roc_auc:.4f}")
    print(f"Candidate Test Recall:  {results['candidate_test_metrics'].recall:.4f}")
    print(f"Candidate Test F1:      {results['candidate_test_metrics'].f1_score:.4f}")
    print(f"Quality Gates:   {'PASSED' if results['quality_gates'].passed else 'FAILED'}")
    print(f"Artifacts saved to: {DEFAULT_CONFIG.artifacts_dir.resolve()}")
    print(f"Reports saved to:   {DEFAULT_CONFIG.reports_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
