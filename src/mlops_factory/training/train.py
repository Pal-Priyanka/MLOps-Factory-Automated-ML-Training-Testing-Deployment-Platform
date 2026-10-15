"""Training orchestrator, baseline vs candidate evaluation, artifact packaging, and verification."""

import hashlib
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mlops_factory.config import DEFAULT_CONFIG, PipelineConfig
from mlops_factory.data.discovery import DatasetMetadata, compute_file_sha256, discover_dataset, inspect_dataset_provenance
from mlops_factory.data.validator import DataValidationResult, validate_dataset
from mlops_factory.evaluation.evaluator import ModelEvaluationResult, evaluate_model_performance
from mlops_factory.evaluation.gates import QualityGateResult, evaluate_quality_gates
from mlops_factory.features.preprocessor import (
    build_preprocessor_pipeline,
    clean_inference_features,
    prepare_features_and_target,
)


class TrainingOrchestrator:
    """Manages the full end-to-end model training, validation, gating, and artifact packaging."""

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.config = config
        self.artifacts_dir = Path(config.artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = Path(config.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        dataset_path: Optional[Path] = None,
        save_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end training workflow:
        1. Discover & fingerprint dataset
        2. Validate schema & audit leakage
        3. Train/Val/Test stratified splits
        4. Fit Baseline and Candidate pipelines
        5. Evaluate on Validation set for model selection
        6. Evaluate winner on holdout Test set
        7. Execute Model Quality Gates
        8. Package artifacts and metadata
        """
        # 1. Dataset discovery & provenance
        if dataset_path is None:
            dataset_path = discover_dataset(self.config.raw_data_filename)

        dataset_meta = inspect_dataset_provenance(dataset_path)

        # 2. Data validation
        raw_df = pd.read_csv(dataset_path)
        val_result = validate_dataset(raw_df, config=self.config)
        if not val_result.is_valid:
            raise ValueError(f"Dataset validation failed: {val_result.validation_errors}")

        # 3. Leakage-safe feature extraction
        X, y = prepare_features_and_target(raw_df, config=self.config)

        # Stratified splits: First Train+Val (80%) vs Test (20%)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=y,
        )

        # Then Train (80% of 80% = 64% total) vs Val (20% of 80% = 16% total)
        val_relative_size = self.config.val_size / (1.0 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_relative_size,
            random_state=self.config.random_seed,
            stratify=y_train_val,
        )

        # 4. Build Pipelines
        preprocessor = build_preprocessor_pipeline(
            numeric_features=self.config.numeric_features,
            categorical_features=self.config.categorical_features,
        )

        # Baseline: Logistic Regression with balanced class weights
        baseline_model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=self.config.random_seed,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        )

        # Candidate: HistGradientBoostingClassifier with balanced class weights
        candidate_model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        random_state=self.config.random_seed,
                        class_weight="balanced",
                        max_iter=150,
                        min_samples_leaf=20,
                        learning_rate=0.08,
                    ),
                ),
            ]
        )

        # 5. Training and Validation Evaluation
        start_t = time.perf_counter()
        baseline_model.fit(X_train, y_train)
        baseline_train_time = time.perf_counter() - start_t
        baseline_val_metrics = evaluate_model_performance(
            baseline_model,
            X_val,
            y_val,
            model_name="Baseline_LogisticRegression",
            dataset_split="validation",
            training_duration=baseline_train_time,
        )

        start_t = time.perf_counter()
        candidate_model.fit(X_train, y_train)
        candidate_train_time = time.perf_counter() - start_t
        candidate_val_metrics = evaluate_model_performance(
            candidate_model,
            X_val,
            y_val,
            model_name="Candidate_HistGradientBoosting",
            dataset_split="validation",
            training_duration=candidate_train_time,
        )

        # Model selection based on validation ROC-AUC
        if candidate_val_metrics.roc_auc >= baseline_val_metrics.roc_auc:
            selected_model = candidate_model
            selected_model_name = "Candidate_HistGradientBoosting"
            selected_train_time = candidate_train_time
        else:
            selected_model = baseline_model
            selected_model_name = "Baseline_LogisticRegression"
            selected_train_time = baseline_train_time

        # 6. Evaluate both on Unbiased Holdout Test Set
        baseline_test_metrics = evaluate_model_performance(
            baseline_model,
            X_test,
            y_test,
            model_name="Baseline_LogisticRegression",
            dataset_split="test",
            training_duration=baseline_train_time,
        )

        candidate_test_metrics = evaluate_model_performance(
            candidate_model,
            X_test,
            y_test,
            model_name="Candidate_HistGradientBoosting",
            dataset_split="test",
            training_duration=candidate_train_time,
        )

        selected_test_metrics = (
            candidate_test_metrics
            if selected_model_name == "Candidate_HistGradientBoosting"
            else baseline_test_metrics
        )

        # 7. Quality Gates Evaluation
        gate_result = evaluate_quality_gates(
            candidate_metrics=candidate_test_metrics,
            gate_config=self.config.quality_gates,
            baseline_metrics=baseline_test_metrics,
        )

        # Reference baseline stats for drift monitoring
        reference_stats = {
            "numeric_means": X_train[self.config.numeric_features].mean().to_dict(),
            "numeric_stds": X_train[self.config.numeric_features].std().to_dict(),
            "numeric_quantiles": {
                col: X_train[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
                for col in self.config.numeric_features
            },
            "train_row_count": len(X_train),
            "val_row_count": len(X_val),
            "test_row_count": len(X_test),
        }

        # 8. Package Artifacts
        artifacts = {}
        if save_artifacts:
            # Retrain selected pipeline on full Train+Val set for maximum operational effectiveness
            final_pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor_pipeline(
                        self.config.numeric_features,
                        self.config.categorical_features,
                    )),
                    (
                        "classifier",
                        (
                            HistGradientBoostingClassifier(
                                random_state=self.config.random_seed,
                                class_weight="balanced",
                                max_iter=150,
                                min_samples_leaf=20,
                                learning_rate=0.08,
                            )
                            if selected_model_name == "Candidate_HistGradientBoosting"
                            else LogisticRegression(
                                max_iter=1000,
                                random_state=self.config.random_seed,
                                class_weight="balanced",
                            )
                        ),
                    ),
                ]
            )
            final_pipeline.fit(X_train_val, y_train_val)

            model_file = self.artifacts_dir / "model.joblib"
            joblib.dump(final_pipeline, model_file)
            model_sha256 = compute_file_sha256(model_file)

            metadata = {
                "model_name": selected_model_name,
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_sha256": model_sha256,
                "dataset_provenance": dataset_meta.to_dict(),
                "dataset_validation": {
                    "is_valid": val_result.is_valid,
                    "row_count": val_result.row_count,
                    "target_distribution": val_result.target_distribution,
                },
                "environment": {
                    "python_version": platform.python_version(),
                    "scikit_learn_version": sklearn.__version__,
                    "platform": platform.platform(),
                },
                "features": {
                    "numeric_features": self.config.numeric_features,
                    "categorical_features": self.config.categorical_features,
                    "total_features": len(self.config.numeric_features) + len(self.config.categorical_features),
                },
                "target_column": self.config.target_column,
                "quality_gates": gate_result.to_dict(),
                "reference_statistics": reference_stats,
            }

            metadata_file = self.artifacts_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            metrics_data = {
                "selected_model": selected_model_name,
                "baseline_test": baseline_test_metrics.to_dict(),
                "candidate_test": candidate_test_metrics.to_dict(),
                "quality_gate_passed": gate_result.passed,
            }

            metrics_file = self.artifacts_dir / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics_data, f, indent=2)

            data_summary_file = self.artifacts_dir / "data_summary.json"
            with open(data_summary_file, "w") as f:
                json.dump(val_result.to_dict(), f, indent=2, default=str)

            artifacts = {
                "model_path": str(model_file),
                "metadata_path": str(metadata_file),
                "metrics_path": str(metrics_file),
                "data_summary_path": str(data_summary_file),
            }

        return {
            "dataset_metadata": dataset_meta,
            "validation_result": val_result,
            "baseline_val_metrics": baseline_val_metrics,
            "candidate_val_metrics": candidate_val_metrics,
            "baseline_test_metrics": baseline_test_metrics,
            "candidate_test_metrics": candidate_test_metrics,
            "selected_model_name": selected_model_name,
            "quality_gates": gate_result,
            "artifacts": artifacts,
        }


def load_and_validate_model_artifact(
    artifacts_dir: Path = Path("artifacts"),
    verify_checksum: bool = True,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Safely loads serialized pipeline and metadata with integrity validation.
    Prevents loading tampered or corrupted artifacts.
    """
    model_path = artifacts_dir / "model.joblib"
    meta_path = artifacts_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at '{model_path}'. "
            f"Please run the training pipeline first: python scripts/run_pipeline.py"
        )

    if not meta_path.exists():
        raise FileNotFoundError(f"Model metadata manifest not found at '{meta_path}'.")

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    if verify_checksum:
        current_sha256 = compute_file_sha256(model_path)
        expected_sha256 = metadata.get("model_sha256")
        if expected_sha256 and current_sha256 != expected_sha256:
            raise ValueError(
                f"Model artifact checksum mismatch! File may be corrupted or modified. "
                f"Expected: {expected_sha256}, Found: {current_sha256}"
            )

    pipeline = joblib.load(model_path)

    # Sanity check pipeline steps
    if not hasattr(pipeline, "predict") or not hasattr(pipeline, "named_steps"):
        raise TypeError("Loaded artifact does not conform to the expected scikit-learn Pipeline interface.")

    return pipeline, metadata
