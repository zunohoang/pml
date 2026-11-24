import csv
import json

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts import evaluate, train


@pytest.fixture
def synthetic_data():
    return {
        "features": np.arange(6 * 403, dtype=float).reshape(6, 403),
        "label_ids": np.array([0, 0, 0, 1, 2, 2]),
        "image_ids": np.arange(6),
        "label_names": np.array(["healthy", "leaf_miner", "rust"]),
        "feature_names": np.array([f"feature_{number}" for number in range(403)]),
    }


def build_candidate(candidate_name, prediction, cv_macro_f1, synthetic_data):
    model_name, class_weight = candidate_name
    classifier = DummyClassifier(strategy="constant", constant=prediction)
    pipeline = make_pipeline(StandardScaler(), classifier)
    pipeline.fit(synthetic_data["features"], synthetic_data["label_ids"])
    return {
        "model_name": model_name, "class_weight": class_weight, "best_estimator": pipeline, "best_params": {},
        "train_accuracy": 0.98765, "cv_accuracy": cv_macro_f1, "cv_accuracy_std": 0.01,
        "train_macro_f1": 0.91234, "best_cv_macro_f1": cv_macro_f1, "cv_std": 0.01, "fit_seconds": 0.1,
    }


@pytest.fixture
def model_dir(tmp_path, synthetic_data):
    directory = tmp_path / "artifacts"
    candidates = []
    for position, (candidate_name, prediction) in enumerate(zip(train.CANDIDATE_OPTIONS, (0, 1, 2, 0))):
        cv_macro_f1 = 0.80 + position / 100
        candidate = build_candidate(candidate_name, prediction, cv_macro_f1, synthetic_data)
        candidates.append(candidate)
    train.save_candidate_models(candidates, directory)
    train.write_cv_summary(list(reversed(candidates)), directory)
    selected_candidate = candidates[-1]
    train.save_best_model(selected_candidate, directory, synthetic_data)
    return directory


def load_synthetic_archive(tmp_path, synthetic_data, model_dir):
    archive_path = tmp_path / "synthetic.npz"
    np.savez_compressed(archive_path, **synthetic_data)
    label_by_code = {0: "healthy", 1: "leaf_miner", 2: "rust"}
    model_info = evaluate.load_best_model_info(model_dir)
    return evaluate.load_test_data(archive_path, label_by_code, model_info)


def test_load_test_data_preserves_arrays(tmp_path, synthetic_data, model_dir):
    loaded_data = load_synthetic_archive(tmp_path, synthetic_data, model_dir)
    for array_name in synthetic_data:
        np.testing.assert_array_equal(loaded_data[array_name], synthetic_data[array_name])


@pytest.mark.parametrize(("array_name", "invalid_value", "error_message"), [
    ("features", np.full((6, 403), np.nan), "finite real numbers"),
    ("features", np.zeros((6, 402)), "403 columns"),
    ("feature_names", np.array(["only_one"]), "one name per feature column"),
    ("image_ids", np.arange(5), "one value per feature row"),
    ("label_names", np.array(["rust", "healthy", "leaf_miner"]), "label_by_code"),
])
def test_load_test_data_rejects_invalid_arrays(
    tmp_path, synthetic_data, model_dir, array_name, invalid_value, error_message
):
    invalid_data = dict(synthetic_data, **{array_name: invalid_value})
    with pytest.raises(ValueError, match=error_message):
        load_synthetic_archive(tmp_path, invalid_data, model_dir)


def test_load_test_data_checks_saved_model_class_names(tmp_path, synthetic_data, model_dir):
    model_info = evaluate.load_best_model_info(model_dir)
    model_info["class_names"] = list(reversed(model_info["class_names"]))
    info_path = model_dir / "best_model_info.json"
    info_path.write_text(json.dumps(model_info), encoding="utf-8")
    with pytest.raises(ValueError, match="class_names in best_model_info.json"):
        load_synthetic_archive(tmp_path, synthetic_data, model_dir)


@pytest.mark.parametrize(("invalid_csv", "error_message"), [
    ("missing_column", "missing required columns"),
    ("missing_row", "exactly four candidate rows"),
    ("duplicate_candidate", "each candidate exactly once"),
])
def test_score_candidates_rejects_invalid_cv_report(synthetic_data, model_dir, invalid_csv, error_message):
    cv_path = model_dir / "cv_results.csv"
    with cv_path.open(newline="", encoding="utf-8") as cv_file:
        rows = list(csv.reader(cv_file))
    if invalid_csv == "missing_column":
        rows[0].remove("train_accuracy")
    elif invalid_csv == "missing_row":
        rows.pop()
    else:
        rows[1] = rows[2]
    with cv_path.open("w", newline="", encoding="utf-8") as cv_file:
        csv.writer(cv_file).writerows(rows)
    with pytest.raises(ValueError, match=error_message):
        evaluate.score_candidates(synthetic_data, ("decision_tree", "balanced"), model_dir)


def test_save_candidate_models_uses_all_four_expected_names(model_dir):
    filenames = {path.name for path in (model_dir / "candidates").iterdir()}
    assert filenames == {
        "svm_none.joblib", "svm_balanced.joblib", "decision_tree_none.joblib", "decision_tree_balanced.joblib",
    }


def test_score_candidates_returns_four_finite_scores_in_candidate_order(synthetic_data, model_dir):
    model_info = evaluate.load_best_model_info(model_dir)
    selected_candidate = evaluate.load_selected_candidate_name(model_info)
    results = evaluate.score_candidates(synthetic_data, selected_candidate, model_dir)
    assert len(results) == 4
    for result in results:
        assert np.isfinite([result["test_accuracy"], result["test_macro_f1"]]).all()
    assert [result["cv_macro_f1"] for result in results] == [0.80, 0.81, 0.82, 0.83]
    assert [result["test_accuracy"] for result in results] == [0.5, 1 / 6, 2 / 6, 0.5]
    assert [result["selected_by_cv"] for result in results] == [False, False, False, True]


def test_selected_candidate_stays_fixed_when_test_scores_change(synthetic_data, model_dir):
    model_info = evaluate.load_best_model_info(model_dir)
    selected_candidate = evaluate.load_selected_candidate_name(model_info)
    original_results = evaluate.score_candidates(synthetic_data, selected_candidate, model_dir)
    changed_data = dict(synthetic_data, label_ids=np.array([1, 1, 1, 1, 0, 2]))
    changed_results = evaluate.score_candidates(changed_data, selected_candidate, model_dir)
    assert selected_candidate == ("decision_tree", "balanced")
    assert changed_results[1]["test_accuracy"] > changed_results[3]["test_accuracy"]
    assert changed_results[3]["test_accuracy"] != original_results[3]["test_accuracy"]
    assert [result["selected_by_cv"] for result in changed_results] == [False, False, False, True]


def test_write_test_results_has_four_rows_and_one_cv_selection(synthetic_data, model_dir):
    results = evaluate.score_candidates(synthetic_data, ("decision_tree", "balanced"), model_dir)
    evaluate.write_test_results(results, model_dir)
    results_path = model_dir / "test_results.csv"
    with results_path.open(newline="", encoding="utf-8") as results_file:
        reader = csv.DictReader(results_file)
        rows = list(reader)
    assert reader.fieldnames == [
        "model_name", "class_weight", "selected_by_cv", "train_accuracy", "cv_accuracy", "test_accuracy",
        "train_macro_f1", "cv_macro_f1", "test_macro_f1",
    ]
    assert len(rows) == 4
    assert [row["selected_by_cv"] for row in rows] == ["False", "False", "False", "True"]
    assert rows[3]["class_weight"] == "balanced"
    assert rows[3]["train_accuracy"] == "0.9877"
    assert rows[3]["cv_macro_f1"] == "0.8300"
    assert rows[3]["test_accuracy"] == "0.5000"


def test_write_test_diagnostics_counts_selected_predictions(synthetic_data, model_dir):
    results = evaluate.score_candidates(synthetic_data, ("svm", "balanced"), model_dir)
    confusion_counts = evaluate.write_test_diagnostics(results, synthetic_data, model_dir)
    assert confusion_counts.sum() == synthetic_data["label_ids"].size
    np.testing.assert_array_equal(confusion_counts, [[0, 3, 0], [0, 1, 0], [0, 2, 0]])
    confusion_path = model_dir / "test_confusion_matrix.csv"
    with confusion_path.open(newline="", encoding="utf-8") as confusion_file:
        confusion_rows = list(csv.reader(confusion_file))
    assert confusion_rows == [
        ["actual_class", "healthy", "leaf_miner", "rust"],
        ["healthy", "0", "3", "0"], ["leaf_miner", "0", "1", "0"], ["rust", "0", "2", "0"],
    ]
    per_class_path = model_dir / "test_per_class.csv"
    with per_class_path.open(newline="", encoding="utf-8") as per_class_file:
        per_class_rows = list(csv.reader(per_class_file))
    assert per_class_rows[0] == ["class_name", "precision", "recall", "f1", "support"]
    assert len(per_class_rows) == 4 and per_class_rows[2] == ["leaf_miner", "0.1667", "1.0000", "0.2857", "1"]


def test_print_test_summary_shows_four_candidates_and_selected_matrix(synthetic_data, model_dir, capsys):
    results = evaluate.score_candidates(synthetic_data, ("decision_tree", "balanced"), model_dir)
    confusion_counts = evaluate.write_test_diagnostics(results, synthetic_data, model_dir)
    evaluate.print_test_summary(results, synthetic_data["label_names"], confusion_counts)
    output_lines = capsys.readouterr().out.splitlines()
    assert output_lines[0].split() == [
        "model", "class_weight", "selected", "train_acc", "val_acc", "test_acc", "train_f1", "val_f1", "test_f1",
    ]
    assert output_lines[5] == "Test confusion matrix (selected by CV)"
    assert [line.split()[2] for line in output_lines[1:5]] == ["False", "False", "False", "True"]
    assert output_lines[6].split() == ["actual", "/", "predicted", "healthy", "leaf_miner", "rust"]
