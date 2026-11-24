import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from coffee_leaf.config import load_config
from coffee_leaf.evaluation.metrics import (
    CV_SCORE_COLUMNS,
    TEST_RESULT_COLUMNS,
    build_confusion_matrix_table,
    build_per_class_table,
    compute_accuracy_and_macro_f1,
    format_test_result_row,
    read_cv_score_row,
)
from coffee_leaf.models.persistence import load_model

if __package__:
    from . import train
else:
    import train


TEST_DATA_PATH = Path("data/processed/features_test.npz")
TEST_RESULTS_FILENAME = "test_results.csv"
TEST_CONFUSION_MATRIX_FILENAME = "test_confusion_matrix.csv"
TEST_PER_CLASS_FILENAME = "test_per_class.csv"

def load_best_model_info(model_dir):
    info_path = Path(model_dir) / train.BEST_MODEL_INFO_FILENAME
    with info_path.open(encoding="utf-8") as info_file:
        model_info = json.load(info_file)
    if not isinstance(model_info, dict):
        raise ValueError("best_model_info.json must contain a JSON object.")
    return model_info


def validate_test_label_names(label_names, label_by_code, model_info):
    label_ids = list(range(label_names.size))
    if sorted(label_by_code) != label_ids:
        raise ValueError("label_by_code IDs must match the indices of test label_names.")
    expected_names = [label_by_code[label_id] for label_id in label_ids]
    test_class_names = label_names.tolist()
    if test_class_names != expected_names:
        raise ValueError("Test label_names must match label_by_code in label ID order.")
    if test_class_names != model_info.get("class_names"):
        raise ValueError("Test label_names must match class_names in best_model_info.json.")


def load_test_data(path, label_by_code, model_info):
    with np.load(path, allow_pickle=False) as archive:
        missing_keys = set(train.TRAINING_ARRAY_NAMES) - set(archive.files)
        if missing_keys:
            raise ValueError(f"Test archive is missing keys: {sorted(missing_keys)}.")
        test_data = {array_name: archive[array_name] for array_name in train.TRAINING_ARRAY_NAMES}
    features = test_data["features"]
    train.validate_training_features(features, test_data["feature_names"])
    train.validate_training_sample_arrays(
        test_data["label_ids"], test_data["image_ids"], test_data["label_names"], features.shape[0]
    )
    validate_test_label_names(test_data["label_names"], label_by_code, model_info)
    return test_data


def load_selected_candidate_name(model_info):
    if "model_name" not in model_info or "class_weight" not in model_info:
        raise ValueError("best_model_info.json must contain model_name and class_weight.")
    candidate_name = model_info["model_name"], model_info["class_weight"]
    if candidate_name not in train.CANDIDATE_OPTIONS:
        raise ValueError("best_model_info.json does not identify a known candidate.")
    return candidate_name


def load_cv_scores(model_dir):
    cv_path = Path(model_dir) / train.CV_SUMMARY_FILENAME
    required_columns = {"model_name", "class_weight", *CV_SCORE_COLUMNS}
    with cv_path.open(newline="", encoding="utf-8") as cv_file:
        reader = csv.DictReader(cv_file)
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError("cv_results.csv is missing required columns.")
        rows = list(reader)
    if len(rows) != len(train.CANDIDATE_OPTIONS):
        raise ValueError("cv_results.csv must contain exactly four candidate rows.")
    scores = {}
    for row in rows:
        weight_name = row["class_weight"]
        if weight_name not in ("none", "balanced"):
            raise ValueError("cv_results.csv contains an invalid class_weight value.")
        class_weight = None if weight_name == "none" else weight_name
        candidate_name = row["model_name"], class_weight
        scores[candidate_name] = read_cv_score_row(row)
    if set(scores) != set(train.CANDIDATE_OPTIONS):
        raise ValueError("cv_results.csv must identify each candidate exactly once.")
    return scores


def score_candidate(model_path, features, label_ids, class_ids):
    model = load_model(model_path)
    predictions = model.predict(features)
    if predictions.shape != label_ids.shape or not np.isin(predictions, class_ids).all():
        raise ValueError(f"Invalid predicted labels from {model_path.name}.")
    accuracy, macro_f1 = compute_accuracy_and_macro_f1(label_ids, predictions, class_ids)
    return {
        "predictions": predictions,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
    }


def score_candidates(test_data, selected_candidate_name, model_dir):
    cv_scores = load_cv_scores(model_dir)
    features = test_data["features"]
    label_ids = test_data["label_ids"]
    class_ids = np.arange(test_data["label_names"].size)
    results = []
    for candidate_name in train.CANDIDATE_OPTIONS:
        model_name, class_weight = candidate_name
        model_path = train.build_candidate_model_path(model_dir, model_name, class_weight)
        test_scores = score_candidate(model_path, features, label_ids, class_ids)
        results.append({
            "model_name": model_name,
            "class_weight": class_weight or "none",
            "selected_by_cv": candidate_name == selected_candidate_name,
            **cv_scores[candidate_name],
            **test_scores,
        })
    return results


def write_test_results(results, model_dir):
    model_directory = Path(model_dir)
    with TemporaryDirectory(dir=model_directory) as temporary_directory:
        staged_path = Path(temporary_directory) / TEST_RESULTS_FILENAME
        with staged_path.open("w", newline="", encoding="utf-8") as results_file:
            writer = csv.DictWriter(results_file, fieldnames=TEST_RESULT_COLUMNS)
            writer.writeheader()
            for result in results:
                writer.writerow(format_test_result_row(result))
        staged_path.replace(model_directory / TEST_RESULTS_FILENAME)


def write_test_diagnostics(results, test_data, model_dir):
    selected_result = next(result for result in results if result["selected_by_cv"])
    label_ids = test_data["label_ids"]
    class_names = test_data["label_names"]
    class_ids = np.arange(class_names.size)
    predictions = selected_result["predictions"]
    confusion_table = build_confusion_matrix_table(label_ids, predictions, class_ids, class_names)
    per_class_table = build_per_class_table(label_ids, predictions, class_ids, class_names)
    model_directory = Path(model_dir)
    with TemporaryDirectory(dir=model_directory) as temporary_directory:
        staged_directory = Path(temporary_directory)
        confusion_path = staged_directory / TEST_CONFUSION_MATRIX_FILENAME
        per_class_path = staged_directory / TEST_PER_CLASS_FILENAME
        train.write_confusion_matrix_file(confusion_path, confusion_table)
        train.write_per_class_file(per_class_path, per_class_table)
        confusion_path.replace(model_directory / TEST_CONFUSION_MATRIX_FILENAME)
        per_class_path.replace(model_directory / TEST_PER_CLASS_FILENAME)
    return confusion_table.to_numpy()


def print_test_summary(results, class_names, confusion_counts):
    print(
        f"{'model':<16} {'class_weight':<13} {'selected':>8} {'train_acc':>9} {'val_acc':>8} {'test_acc':>8} "
        f"{'train_f1':>8} {'val_f1':>8} {'test_f1':>8}"
    )
    for result in results:
        selected = str(result["selected_by_cv"])
        print(
            f"{result['model_name']:<16} {result['class_weight']:<13} {selected:>8} "
            f"{result['train_accuracy']:>9.3f} {result['cv_accuracy']:>8.3f} {result['test_accuracy']:>8.3f} "
            f"{result['train_macro_f1']:>8.3f} {result['cv_macro_f1']:>8.3f} {result['test_macro_f1']:>8.3f}"
        )
    print("Test confusion matrix (selected by CV)")
    train.print_confusion_matrix(class_names, confusion_counts)


def main():
    configuration = load_config(train.CONFIG_PATH)
    model_dir = configuration["model_dir"]
    model_info = load_best_model_info(model_dir)
    test_data = load_test_data(TEST_DATA_PATH, configuration["label_by_code"], model_info)
    selected_candidate_name = load_selected_candidate_name(model_info)
    results = score_candidates(test_data, selected_candidate_name, model_dir)
    write_test_results(results, model_dir)
    confusion_counts = write_test_diagnostics(results, test_data, model_dir)
    print_test_summary(results, test_data["label_names"], confusion_counts)


if __name__ == "__main__":
    main()
