import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


CV_METRIC_COLUMNS = (
    "train_accuracy", "cv_accuracy", "cv_accuracy_std", "train_macro_f1", "best_cv_macro_f1", "cv_std", "fit_seconds",
)
CV_SCORE_RESULT_KEYS = (
    ("train_accuracy", "mean_train_accuracy"),
    ("cv_accuracy", "mean_test_accuracy"),
    ("cv_accuracy_std", "std_test_accuracy"),
    ("train_macro_f1", "mean_train_f1_macro"),
    ("cv_std", "std_test_f1_macro"),
)

CV_SCORE_COLUMNS = ("train_accuracy", "cv_accuracy", "train_macro_f1", "best_cv_macro_f1")
TEST_SCORE_COLUMNS = ("train_accuracy", "cv_accuracy", "test_accuracy", "train_macro_f1", "cv_macro_f1", "test_macro_f1")
TEST_RESULT_COLUMNS = ("model_name", "class_weight", "selected_by_cv") + TEST_SCORE_COLUMNS


def summarize_candidate(model_name, class_weight, search, fit_seconds):
    best_index = search.best_index_
    metrics = {"best_cv_macro_f1": float(search.best_score_), "fit_seconds": float(fit_seconds)}
    for metric_name, result_key in CV_SCORE_RESULT_KEYS:
        result_values = search.cv_results_[result_key]
        metrics[metric_name] = float(result_values[best_index])
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError(f"{model_name} returned nonfinite training metrics.")
    return {
        "model_name": model_name,
        "class_weight": class_weight,
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        **metrics,
    }


def format_cv_summary_row(candidate):
    class_weight = candidate["class_weight"]
    if class_weight is None:
        class_weight = "none"
    summary_row = {
        "model_name": candidate["model_name"],
        "class_weight": class_weight,
        "best_params": json.dumps(candidate["best_params"], sort_keys=True, allow_nan=False),
    }
    for metric_name in CV_METRIC_COLUMNS:
        summary_row[metric_name] = f"{candidate[metric_name]:.4f}"
    return summary_row


def get_candidate_rank(candidate):
    return candidate["best_cv_macro_f1"], -candidate["cv_std"]


def select_best_candidate(candidates):
    if not candidates:
        raise ValueError("At least one candidate is required.")
    return max(candidates, key=get_candidate_rank)


def read_cv_score_row(row):
    scores = {}
    for column_name in CV_SCORE_COLUMNS:
        try:
            scores[column_name] = float(row[column_name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"cv_results.csv contains an invalid {column_name} value.") from error
    if not np.isfinite(list(scores.values())).all():
        raise ValueError("cv_results.csv must contain finite scores.")
    scores["cv_macro_f1"] = scores.pop("best_cv_macro_f1")
    return scores


def format_test_result_row(result):
    row = {column_name: result[column_name] for column_name in TEST_RESULT_COLUMNS}
    for score_name in TEST_SCORE_COLUMNS:
        row[score_name] = f"{result[score_name]:.4f}"
    return row


def compute_accuracy_and_macro_f1(true_label_ids, predicted_label_ids, class_ids):
    accuracy = accuracy_score(true_label_ids, predicted_label_ids)
    macro_f1 = f1_score(true_label_ids, predicted_label_ids, labels=class_ids, average="macro", zero_division=0)
    return accuracy, macro_f1


def build_confusion_matrix_table(true_label_ids, predicted_label_ids, class_ids, class_names):
    confusion_counts = confusion_matrix(true_label_ids, predicted_label_ids, labels=class_ids)
    actual_classes = pd.Index(class_names, name="actual_class")
    return pd.DataFrame(confusion_counts, index=actual_classes, columns=class_names)


def build_per_class_table(true_label_ids, predicted_label_ids, class_ids, class_names):
    precision, recall, class_f1, support = precision_recall_fscore_support(
        true_label_ids, predicted_label_ids, labels=class_ids, zero_division=0,
    )
    return pd.DataFrame({
        "class_name": class_names,
        "precision": precision,
        "recall": recall,
        "f1": class_f1,
        "support": support,
    })
