"""Automated model-quality gates and release acceptance criteria."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from mlops_factory.config import QualityGateConfig
from mlops_factory.evaluation.evaluator import ModelEvaluationResult


@dataclass
class GateCheck:
    metric_name: str
    observed_value: float
    threshold_value: float
    comparison_operator: str  # '>=' or '<='
    passed: bool
    description: str


@dataclass
class QualityGateResult:
    """Evaluation outcome of automated model-quality gates."""
    passed: bool
    model_name: str
    checks: List[GateCheck]
    passed_checks_count: int
    total_checks_count: int
    failure_reasons: List[str] = field(default_factory=list)
    is_production_ready: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "model_name": self.model_name,
            "passed_checks_count": self.passed_checks_count,
            "total_checks_count": self.total_checks_count,
            "failure_reasons": self.failure_reasons,
            "is_production_ready": self.is_production_ready,
            "checks": [asdict(c) for c in self.checks],
        }


def evaluate_quality_gates(
    candidate_metrics: ModelEvaluationResult,
    gate_config: QualityGateConfig,
    baseline_metrics: Optional[ModelEvaluationResult] = None,
) -> QualityGateResult:
    """
    Applies objective, configurable acceptance thresholds to candidate model metrics.
    Ensures that only models exceeding performance baselines can be promoted to production.
    """
    checks: List[GateCheck] = []
    failure_reasons: List[str] = []

    # 1. ROC-AUC gate
    passed_auc = candidate_metrics.roc_auc >= gate_config.min_roc_auc
    checks.append(
        GateCheck(
            metric_name="ROC-AUC",
            observed_value=candidate_metrics.roc_auc,
            threshold_value=gate_config.min_roc_auc,
            comparison_operator=">=",
            passed=passed_auc,
            description="Discrimination capability across classification thresholds",
        )
    )
    if not passed_auc:
        failure_reasons.append(
            f"ROC-AUC {candidate_metrics.roc_auc} below gate threshold {gate_config.min_roc_auc}"
        )

    # 2. Recall gate (Crucial for churn to identify high-risk accounts)
    passed_rec = candidate_metrics.recall >= gate_config.min_recall
    checks.append(
        GateCheck(
            metric_name="Recall (Churners)",
            observed_value=candidate_metrics.recall,
            threshold_value=gate_config.min_recall,
            comparison_operator=">=",
            passed=passed_rec,
            description="Proportion of actual churners identified by the model",
        )
    )
    if not passed_rec:
        failure_reasons.append(
            f"Recall {candidate_metrics.recall} below gate threshold {gate_config.min_recall}"
        )

    # 3. Precision gate (Avoid inundating retention team with false alarms)
    passed_prec = candidate_metrics.precision >= gate_config.min_precision
    checks.append(
        GateCheck(
            metric_name="Precision (Churners)",
            observed_value=candidate_metrics.precision,
            threshold_value=gate_config.min_precision,
            comparison_operator=">=",
            passed=passed_prec,
            description="Ratio of true churners among all predicted churners",
        )
    )
    if not passed_prec:
        failure_reasons.append(
            f"Precision {candidate_metrics.precision} below gate threshold {gate_config.min_precision}"
        )

    # 4. F1-Score gate
    passed_f1 = candidate_metrics.f1_score >= gate_config.min_f1_score
    checks.append(
        GateCheck(
            metric_name="F1-Score",
            observed_value=candidate_metrics.f1_score,
            threshold_value=gate_config.min_f1_score,
            comparison_operator=">=",
            passed=passed_f1,
            description="Harmonic mean of precision and recall",
        )
    )
    if not passed_f1:
        failure_reasons.append(
            f"F1-Score {candidate_metrics.f1_score} below gate threshold {gate_config.min_f1_score}"
        )

    # 5. Accuracy gate
    passed_acc = candidate_metrics.accuracy >= gate_config.min_accuracy
    checks.append(
        GateCheck(
            metric_name="Accuracy",
            observed_value=candidate_metrics.accuracy,
            threshold_value=gate_config.min_accuracy,
            comparison_operator=">=",
            passed=passed_acc,
            description="Overall accuracy across both classes",
        )
    )
    if not passed_acc:
        failure_reasons.append(
            f"Accuracy {candidate_metrics.accuracy} below gate threshold {gate_config.min_accuracy}"
        )

    # 6. Latency / Training duration gate
    passed_duration = candidate_metrics.training_duration_seconds <= gate_config.max_training_duration_seconds
    checks.append(
        GateCheck(
            metric_name="Training Duration (s)",
            observed_value=candidate_metrics.training_duration_seconds,
            threshold_value=gate_config.max_training_duration_seconds,
            comparison_operator="<=",
            passed=passed_duration,
            description="Execution time budget for retraining pipelines",
        )
    )
    if not passed_duration:
        failure_reasons.append(
            f"Training duration {candidate_metrics.training_duration_seconds}s exceeds budget {gate_config.max_training_duration_seconds}s"
        )

    # 7. Comparison with baseline (if provided)
    if baseline_metrics is not None:
        passed_vs_base = candidate_metrics.roc_auc >= (baseline_metrics.roc_auc - 0.02)
        checks.append(
            GateCheck(
                metric_name="Candidate vs Baseline ROC-AUC",
                observed_value=candidate_metrics.roc_auc,
                threshold_value=baseline_metrics.roc_auc,
                comparison_operator=">=",
                passed=passed_vs_base,
                description=f"Candidate ROC-AUC vs Baseline ({baseline_metrics.roc_auc})",
            )
        )
        if not passed_vs_base:
            failure_reasons.append(
                f"Candidate ROC-AUC ({candidate_metrics.roc_auc}) degraded compared to baseline ({baseline_metrics.roc_auc})"
            )

    passed_count = sum(1 for c in checks if c.passed)
    total_count = len(checks)
    all_passed = passed_count == total_count

    return QualityGateResult(
        passed=all_passed,
        model_name=candidate_metrics.model_name,
        checks=checks,
        passed_checks_count=passed_count,
        total_checks_count=total_count,
        failure_reasons=failure_reasons,
        is_production_ready=all_passed,
    )
