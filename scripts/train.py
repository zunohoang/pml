import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from sklearn import __version__ as SKLEARN_VERSION

from coffee_leaf.config import load_config
from coffee_leaf.evaluation.metrics import (
    CV_METRIC_COLUMNS,
    build_confusion_matrix_table,
    build_per_class_table,
    compute_accuracy_and_macro_f1,
    format_cv_summary_row,
    get_candidate_rank,
    select_best_candidate,
    summarize_candidate,
)
from coffee_leaf.models.builders import build_decision_tree_pipeline, build_svm_pipeline
from coffee_leaf.models.persistence import save_model
from coffee_leaf.models.search import build_cv_predictions, build_parameter_grid, run_grid_search


CONFIG_PATH = Path("configs/default.yaml")
TRAINING_DATA_PATH = Path("data/processed/features_train.npz")
TRAINING_ARRAY_NAMES = ("features", "label_ids", "image_ids", "label_names", "feature_names")
EXPECTED_FEATURE_COUNT = 403

CANDIDATE_OPTIONS = (
    ("svm", None), ("svm", "balanced"),
    ("decision_tree", None), ("decision_tree", "balanced"),
)

CV_SUMMARY_COLUMNS = ("model_name", "class_weight", "best_params") + CV_METRIC_COLUMNS

CV_SUMMARY_FILENAME = "cv_results.csv"
CV_CONFUSION_MATRIX_FILENAME = "cv_confusion_matrix.csv"
CV_PER_CLASS_FILENAME = "cv_per_class.csv"
BEST_MODEL_FILENAME = "best_model.joblib"
BEST_MODEL_INFO_FILENAME = "best_model_info.json"


def validate_training_features(features, feature_names):
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"features must have at least one row and {EXPECTED_FEATURE_COUNT} columns.")
    if feature_names.shape != (features.shape[1],):
        raise ValueError("feature_names must contain one name per feature column.")
    if features.dtype.kind not in "iuf" or not np.isfinite(features).all():
        raise ValueError("features must contain finite real numbers.")


def validate_training_sample_arrays(label_ids, image_ids, label_names, row_count):
    if label_ids.shape != (row_count,) or image_ids.shape != (row_count,):
        raise ValueError("label_ids and image_ids must each contain one value per feature row.")
    if label_ids.dtype.kind not in "iu" or image_ids.dtype.kind not in "iu":
        raise ValueError("label_ids and image_ids must contain integers.")
    if label_names.ndim != 1 or label_names.size == 0 or label_names.dtype.kind != "U":
        raise ValueError("label_names must be a nonempty one-dimensional array of strings.")
    if label_ids.min() < 0 or label_ids.max() >= label_names.size:
        raise ValueError("label_ids must be valid indices into label_names.")


def load_training_data(path=TRAINING_DATA_PATH):
    with np.load(path, allow_pickle=False) as archive:
        missing_keys = set(TRAINING_ARRAY_NAMES) - set(archive.files)
        if missing_keys:
            raise ValueError(f"Training archive is missing keys: {sorted(missing_keys)}.")
        training_data = {array_name: archive[array_name] for array_name in TRAINING_ARRAY_NAMES}
    features = training_data["features"]
    validate_training_features(features, training_data["feature_names"])
    validate_training_sample_arrays(
        training_data["label_ids"], training_data["image_ids"], training_data["label_names"], features.shape[0]
    )
    return training_data


def validate_label_names(label_names, label_by_code):
    label_ids = list(range(label_names.size))
    if sorted(label_by_code) != label_ids:
        raise ValueError("label_by_code IDs must match the indices of training label_names.")
    expected_names = [label_by_code[label_id] for label_id in label_ids]
    if label_names.tolist() != expected_names:
        raise ValueError("Training label_names must match label_by_code in label ID order.")


def build_candidate_parameter_grids(configuration):
    return {
        "svm": build_parameter_grid("svm", configuration),
        "decision_tree": build_parameter_grid("decision_tree", configuration),
    }


def train_candidate_model(model_name, class_weight, parameter_grid, training_data, configuration):
    if model_name == "svm":
        pipeline = build_svm_pipeline(class_weight, configuration)
    else:
        pipeline = build_decision_tree_pipeline(class_weight, configuration)
    features = training_data["features"]
    label_ids = training_data["label_ids"]
    image_ids = training_data["image_ids"]
    started = perf_counter()
    search = run_grid_search(pipeline, parameter_grid, features, label_ids, image_ids, configuration)
    fit_seconds = perf_counter() - started
    print(f"{model_name} class_weight={class_weight} fit_seconds={fit_seconds:.4f}", flush=True)
    return summarize_candidate(model_name, class_weight, search, fit_seconds)


def train_candidate_models(training_data, parameter_grids, configuration=None):
    candidates = []
    for model_name, class_weight in CANDIDATE_OPTIONS:
        parameter_grid = parameter_grids[model_name]
        candidate = train_candidate_model(model_name, class_weight, parameter_grid, training_data, configuration)
        candidates.append(candidate)
    return candidates


def write_cv_summary(candidates, model_dir):
    if len(candidates) != len(CANDIDATE_OPTIONS):
        raise ValueError("The CV summary requires four candidates.")
    summary_rows = [format_cv_summary_row(candidate) for candidate in candidates]
    model_directory = Path(model_dir)
    model_directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=model_directory) as temporary_directory:
        staged_report = Path(temporary_directory) / CV_SUMMARY_FILENAME
        with staged_report.open("w", newline="", encoding="utf-8") as report_file:
            writer = csv.DictWriter(report_file, fieldnames=CV_SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(summary_rows)
        staged_report.replace(model_directory / CV_SUMMARY_FILENAME)


def print_training_summary(candidates):
    print(
        f"{'model':<16} {'class_weight':<13} {'train_acc':>9} {'val_acc(train_cv_mean)':>22} "
        f"{'train_f1':>8} {'val_f1(train_cv_mean)':>22} {'gap_f1':>8}"
    )
    for candidate in candidates:
        model_name = candidate["model_name"]
        class_weight = candidate["class_weight"] or "none"
        train_accuracy = candidate["train_accuracy"]
        cv_accuracy = candidate["cv_accuracy"]
        train_macro_f1 = candidate["train_macro_f1"]
        cv_macro_f1 = candidate["best_cv_macro_f1"]
        gap_macro_f1 = train_macro_f1 - cv_macro_f1
        print(
            f"{model_name:<16} {class_weight:<13} {train_accuracy:>9.3f} {cv_accuracy:>22.3f} "
            f"{train_macro_f1:>8.3f} {cv_macro_f1:>22.3f} {gap_macro_f1:>8.3f}"
        )


def write_confusion_matrix_file(path, confusion_table):
    confusion_table.to_csv(path, encoding="utf-8", lineterminator="\r\n")


def write_per_class_file(path, per_class_table):
    per_class_table.to_csv(path, index=False, encoding="utf-8", float_format="%.4f", lineterminator="\r\n")


def save_cv_diagnostic_files(model_dir, confusion_table, per_class_table):
    model_directory = Path(model_dir)
    model_directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=model_directory) as temporary_directory:
        staged_directory = Path(temporary_directory)
        confusion_path = staged_directory / CV_CONFUSION_MATRIX_FILENAME
        per_class_path = staged_directory / CV_PER_CLASS_FILENAME
        write_confusion_matrix_file(confusion_path, confusion_table)
        write_per_class_file(per_class_path, per_class_table)
        confusion_path.replace(model_directory / CV_CONFUSION_MATRIX_FILENAME)
        per_class_path.replace(model_directory / CV_PER_CLASS_FILENAME)


def print_confusion_matrix(class_names, confusion_counts):
    row_heading = "actual / predicted"
    longest_class_name = max(len(class_name) for class_name in class_names)
    row_width = max(len(row_heading), longest_class_name) + 2
    column_width = max(7, longest_class_name) + 2
    column_headings = "".join(f"{class_name:>{column_width}}" for class_name in class_names)
    print(f"{row_heading:<{row_width}}{column_headings}")
    for class_name, row_counts in zip(class_names, confusion_counts):
        columns = "".join(f"{sample_count:>{column_width}}" for sample_count in row_counts)
        print(f"{class_name:<{row_width}}{columns}")


def write_cv_diagnostics(best_candidate, training_data, configuration, model_dir):
    features = training_data["features"]
    label_ids = training_data["label_ids"]
    image_ids = training_data["image_ids"]
    class_names = training_data["label_names"]
    class_ids = np.arange(class_names.size)
    best_pipeline = best_candidate["best_estimator"]
    predictions = build_cv_predictions(best_pipeline, features, label_ids, image_ids, configuration)
    confusion_table = build_confusion_matrix_table(label_ids, predictions, class_ids, class_names)
    per_class_table = build_per_class_table(label_ids, predictions, class_ids, class_names)
    save_cv_diagnostic_files(model_dir, confusion_table, per_class_table)
    accuracy, macro_f1 = compute_accuracy_and_macro_f1(label_ids, predictions, class_ids)
    print(f"Train CV out-of-fold accuracy={accuracy:.4f} macro_f1={macro_f1:.4f}")
    print_confusion_matrix(class_names, confusion_table.to_numpy())


def build_best_model_info(best_candidate, training_data):
    best_model = best_candidate["best_estimator"]
    class_names = training_data["label_names"][best_model.classes_]
    train_rows, feature_count = training_data["features"].shape
    return {
        "model_name": best_candidate["model_name"],
        "class_weight": best_candidate["class_weight"],
        "best_params": best_candidate["best_params"],
        "best_cv_macro_f1": best_candidate["best_cv_macro_f1"],
        "cv_std": best_candidate["cv_std"],
        "train_macro_f1": best_candidate["train_macro_f1"],
        "train_accuracy": best_candidate["train_accuracy"],
        "cv_accuracy": best_candidate["cv_accuracy"],
        "feature_count": feature_count,
        "train_rows": train_rows,
        "class_names": class_names.tolist(),
        "sklearn_version": SKLEARN_VERSION,
    }


def save_best_model(best_candidate, model_dir, training_data):
    model_info = build_best_model_info(best_candidate, training_data)
    info_text = json.dumps(model_info, indent=2, ensure_ascii=False, allow_nan=False)
    model_directory = Path(model_dir)
    model_directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=model_directory) as temporary_directory:
        staged_directory = Path(temporary_directory)
        staged_model = staged_directory / BEST_MODEL_FILENAME
        staged_info = staged_directory / BEST_MODEL_INFO_FILENAME
        save_model(best_candidate["best_estimator"], staged_model)
        staged_info.write_text(info_text + "\n", encoding="utf-8")
        staged_model.replace(model_directory / BEST_MODEL_FILENAME)
        staged_info.replace(model_directory / BEST_MODEL_INFO_FILENAME)


def build_candidate_model_path(model_dir, model_name, class_weight):
    weight_name = class_weight or "none"
    return Path(model_dir) / "candidates" / f"{model_name}_{weight_name}.joblib"


def save_candidate_models(candidates, model_dir):
    model_directory = Path(model_dir)
    model_directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=model_directory) as temporary_directory:
        staged_models = []
        for candidate in candidates:
            model_name, class_weight = candidate["model_name"], candidate["class_weight"]
            staged_path = build_candidate_model_path(temporary_directory, model_name, class_weight)
            model_path = build_candidate_model_path(model_directory, model_name, class_weight)
            save_model(candidate["best_estimator"], staged_path)
            staged_models.append((staged_path, model_path))
        for staged_path, model_path in staged_models:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.replace(model_path)


def main():
    configuration = load_config(CONFIG_PATH)
    training_data = load_training_data()
    validate_label_names(training_data["label_names"], configuration["label_by_code"])
    candidates = train_candidate_models(training_data, build_candidate_parameter_grids(configuration), configuration)
    write_cv_summary(candidates, configuration["model_dir"])
    print_training_summary(candidates)
    best_candidate = select_best_candidate(candidates)
    write_cv_diagnostics(best_candidate, training_data, configuration, configuration["model_dir"])
    save_candidate_models(candidates, configuration["model_dir"])
    save_best_model(best_candidate, configuration["model_dir"], training_data)


if __name__ == "__main__":
    main()
