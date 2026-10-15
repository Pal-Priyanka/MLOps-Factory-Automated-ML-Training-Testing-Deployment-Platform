"""Unit and integration tests for training orchestration, quality gates, and artifact integrity."""

import pytest
from mlops_factory.config import PipelineConfig, QualityGateConfig
from mlops_factory.evaluation.evaluator import (
    ConfusionMatrixSummary,
    ModelEvaluationResult,
    evaluate_model_performance,
)
from mlops_factory.evaluation.gates import evaluate_quality_gates
from mlops_factory.training.train import (
    TrainingOrchestrator,
    load_and_validate_model_artifact,
)


def test_evaluate_model_performance_metrics(fitted_mock_pipeline, sample_synthetic_dataframe):
    """Ensures evaluate_model_performance generates all required classification metrics."""
    from mlops_factory.features.preprocessor import clean_inference_features

    df = sample_synthetic_dataframe
    X = clean_inference_features(df)
    y = (df["Churn Label"] == "Yes").astype(int)

    metrics = evaluate_model_performance(
        fitted_mock_pipeline,
        X,
        y,
        model_name="MockModel",
        dataset_split="test",
        training_duration=0.5,
    )

    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert metrics.confusion_matrix.true_negatives >= 0
    assert metrics.confusion_matrix.true_positives >= 0
    assert "churners_identified_recall" in metrics.business_tradeoff_summary


def test_quality_gates_pass_and_fail_conditions():
    """Validates that quality gates accurately pass compliant models and block sub-standard ones."""
    cm = ConfusionMatrixSummary(800, 100, 50, 250)

    passing_result = ModelEvaluationResult(
        model_name="HighQualityModel",
        dataset_split="test",
        accuracy=0.85,
        precision=0.71,
        recall=0.83,
        f1_score=0.77,
        roc_auc=0.91,
        pr_auc=0.80,
        confusion_matrix=cm,
        class_distribution={"0": 900, "1": 300},
        training_duration_seconds=5.0,
        evaluation_duration_seconds=0.1,
    )

    gate_cfg = QualityGateConfig(
        min_roc_auc=0.82,
        min_recall=0.70,
        min_precision=0.50,
        min_f1_score=0.60,
        min_accuracy=0.75,
        max_training_duration_seconds=60.0,
    )

    pass_eval = evaluate_quality_gates(passing_result, gate_cfg)
    assert pass_eval.passed is True
    assert pass_eval.is_production_ready is True
    assert len(pass_eval.failure_reasons) == 0

    # Sub-standard model (low recall & low ROC-AUC)
    failing_result = ModelEvaluationResult(
        model_name="WeakModel",
        dataset_split="test",
        accuracy=0.68,
        precision=0.45,
        recall=0.40,
        f1_score=0.42,
        roc_auc=0.65,
        pr_auc=0.40,
        confusion_matrix=cm,
        class_distribution={"0": 900, "1": 300},
        training_duration_seconds=5.0,
        evaluation_duration_seconds=0.1,
    )

    fail_eval = evaluate_quality_gates(failing_result, gate_cfg)
    assert fail_eval.passed is False
    assert fail_eval.is_production_ready is False
    assert len(fail_eval.failure_reasons) > 0


def test_training_orchestrator_smoke_test(tmp_path, sample_synthetic_dataframe):
    """Executes a complete, fast training smoke test with custom temporary artifact paths."""
    csv_path = tmp_path / "test_data.csv"
    sample_synthetic_dataframe.to_csv(csv_path, index=False)

    artifacts_path = tmp_path / "artifacts"
    reports_path = tmp_path / "reports"

    cfg = PipelineConfig(
        artifacts_dir=artifacts_path,
        reports_dir=reports_path,
        test_size=0.25,
        val_size=0.20,
        random_seed=42,
    )

    orchestrator = TrainingOrchestrator(config=cfg)
    results = orchestrator.run(dataset_path=csv_path, save_artifacts=True)

    assert results["selected_model_name"] in [
        "Candidate_HistGradientBoosting",
        "Baseline_LogisticRegression",
    ]
    assert (artifacts_path / "model.joblib").exists()
    assert (artifacts_path / "metadata.json").exists()
    assert (artifacts_path / "metrics.json").exists()

    # Test safe loading & checksum verification
    pipeline, meta = load_and_validate_model_artifact(artifacts_dir=artifacts_path)
    assert hasattr(pipeline, "predict")
    assert meta["model_name"] == results["selected_model_name"]


def test_corrupted_model_artifact_triggers_checksum_error(tmp_path, sample_synthetic_dataframe):
    """Verifies that tampering with serialized model bytes triggers SHA-256 validation error."""
    csv_path = tmp_path / "test_data.csv"
    sample_synthetic_dataframe.to_csv(csv_path, index=False)

    artifacts_path = tmp_path / "artifacts"
    cfg = PipelineConfig(artifacts_dir=artifacts_path, reports_dir=tmp_path / "reports")

    orchestrator = TrainingOrchestrator(config=cfg)
    orchestrator.run(dataset_path=csv_path, save_artifacts=True)

    model_file = artifacts_path / "model.joblib"
    # Corrupt model file by appending bytes
    with open(model_file, "ab") as f:
        f.write(b"\x00\x00CORRUPTED_BYTES")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_and_validate_model_artifact(artifacts_dir=artifacts_path, verify_checksum=True)
