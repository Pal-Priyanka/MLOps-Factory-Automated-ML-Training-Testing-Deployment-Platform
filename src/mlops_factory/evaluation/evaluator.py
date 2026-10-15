"""Model performance evaluation, metric calculation, and explainability summaries."""

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


@dataclass
class ConfusionMatrixSummary:
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class ModelEvaluationResult:
    """Comprehensive performance metrics for a trained classifier."""
    model_name: str
    dataset_split: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: ConfusionMatrixSummary
    class_distribution: Dict[str, int]
    training_duration_seconds: float
    evaluation_duration_seconds: float
    top_feature_importances: List[Dict[str, Any]] = field(default_factory=list)
    business_tradeoff_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["confusion_matrix"] = self.confusion_matrix.to_dict()
        return d


def extract_feature_importances(
    pipeline: Pipeline,
    max_features: int = 15,
) -> List[Dict[str, Any]]:
    """Extracts top feature importances from pipeline's classifier."""
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        classifier = pipeline.named_steps.get("classifier")

        if preprocessor is None or classifier is None:
            return []

        # Get transformed feature names
        feature_names = preprocessor.get_feature_names_out()

        importances = None
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            # For linear models, take absolute magnitude of first row
            importances = np.abs(classifier.coef_[0])

        if importances is None:
            return []

        # Pair and sort
        paired = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )

        results = [
            {"feature": name, "importance": round(float(val), 5)}
            for name, val in paired[:max_features]
        ]
        return results
    except Exception:
        return []


def evaluate_model_performance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "CandidateModel",
    dataset_split: str = "test",
    training_duration: float = 0.0,
) -> ModelEvaluationResult:
    """
    Computes rigorous classification metrics on evaluation data.
    """
    start_eval = time.perf_counter()

    y_pred = pipeline.predict(X)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X)[:, 1]
    else:
        # Fallback to decision function or binary predictions
        y_prob = pipeline.decision_function(X)

    eval_duration = round(time.perf_counter() - start_eval, 4)

    # Core metrics
    acc = round(float(accuracy_score(y, y_pred)), 4)
    prec = round(float(precision_score(y, y_pred, zero_division=0)), 4)
    rec = round(float(recall_score(y, y_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y, y_pred, zero_division=0)), 4)

    try:
        roc_auc = round(float(roc_auc_score(y, y_prob)), 4)
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = round(float(average_precision_score(y, y_prob)), 4)
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cm_summary = ConfusionMatrixSummary(
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = {str(k): int(v) for k, v in zip(unique, counts)}

    # Churn business trade-off
    # In telecom churn, FN (missed churners who leave without retention outreach) are expensive,
    # while FP (loyal customers who receive unnecessary discount offers) incur small promotion cost.
    tradeoff = {
        "churners_identified_recall": rec,
        "missed_churners_count": int(fn),
        "false_alarm_promotions_count": int(fp),
        "interpretation": (
            f"Of {int(tp + fn)} actual churners, {int(tp)} were identified ({rec * 100:.1f}%), "
            f"while {int(fn)} churners were missed. There were {int(fp)} false alarms."
        ),
    }

    feature_importances = extract_feature_importances(pipeline)

    return ModelEvaluationResult(
        model_name=model_name,
        dataset_split=dataset_split,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm_summary,
        class_distribution=class_dist,
        training_duration_seconds=round(training_duration, 4),
        evaluation_duration_seconds=eval_duration,
        top_feature_importances=feature_importances,
        business_tradeoff_summary=tradeoff,
    )
