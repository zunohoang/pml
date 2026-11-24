import csv
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from coffee_leaf.config import load_config
from coffee_leaf.models.builders import build_decision_tree_pipeline, build_svm_pipeline
from coffee_leaf.models.persistence import load_model, save_model
from coffee_leaf.models.search import build_cv_predictions, build_parameter_grid, run_grid_search
from scripts import train


def create_synthetic_training_data():
    random_source = np.random.default_rng(42)
    label_ids = np.tile(np.arange(3), 20)
    image_ids = np.repeat(np.arange(20), 3)
    variable_features = random_source.normal(size=(60, 5))
    variable_features[:, 0] += 3 * label_ids
    constant_feature = np.ones(60)
    features = np.column_stack((variable_features, constant_feature))
    return {
        "features": features,
        "label_ids": label_ids,
        "image_ids": image_ids,
        "label_names": np.array(["lime", "amber", "green"]),
    }


def create_small_parameter_grids():
    return {
        "svm": {"svc__C": [0.1, 1.0], "svc__gamma": ["scale"]},
        "decision_tree": {"decisiontreeclassifier__max_depth": [2, 4], "decisiontreeclassifier__min_samples_leaf": [1]},
    }


@pytest.mark.parametrize("build_pipeline, estimator_name", [
    (build_svm_pipeline, "svc"), (build_decision_tree_pipeline, "decisiontreeclassifier"),
])
def test_pipeline_scales_first_and_fits_constant_feature(build_pipeline, estimator_name):
    training_data = create_synthetic_training_data()
    pipeline = build_pipeline("balanced")
    assert isinstance(pipeline.steps[0][1], StandardScaler)
    classifier = pipeline.named_steps[estimator_name]
    assert classifier.class_weight == "balanced"
    configuration = load_config("configs/default.yaml")
    assert classifier.random_state == configuration["seed"]
    pipeline.fit(training_data["features"], training_data["label_ids"])
    if estimator_name == "svc":
        model_scores = pipeline.decision_function(training_data["features"])
    else:
        model_scores = pipeline.predict_proba(training_data["features"])
    assert model_scores.shape == (60, 3)
    assert np.isfinite(model_scores).all()
    with pytest.raises(ValueError, match="class_weight"):
        build_pipeline("invalid")


def test_parameter_grids_match_configuration():
    configuration = load_config("configs/default.yaml")
    svm_grid = build_parameter_grid("svm")
    assert svm_grid == {"svc__C": configuration["svm"]["C"], "svc__gamma": configuration["svm"]["gamma"]}
    tree_grid = build_parameter_grid("decision_tree")
    assert tree_grid == {
        "decisiontreeclassifier__max_depth": configuration["decision_tree"]["max_depth"],
        "decisiontreeclassifier__min_samples_leaf": configuration["decision_tree"]["min_samples_leaf"],
    }
    with pytest.raises(ValueError, match="Unknown model name"):
        build_parameter_grid("invalid")


def test_grid_search_repeats_best_parameters_with_group_folds():
    training_data = create_synthetic_training_data()
    configuration = load_config("configs/default.yaml")
    pipeline = build_svm_pipeline(None, configuration)
    parameter_grid = create_small_parameter_grids()["svm"]
    features = training_data["features"]
    label_ids = training_data["label_ids"]
    image_ids = training_data["image_ids"]
    first_search = run_grid_search(pipeline, parameter_grid, features, label_ids, image_ids, configuration)
    repeated_search = run_grid_search(pipeline, parameter_grid, features, label_ids, image_ids, configuration)
    assert first_search.best_params_ == repeated_search.best_params_
    assert first_search.best_score_ == repeated_search.best_score_
    for parameter_name, chosen_value in first_search.best_params_.items():
        assert chosen_value in parameter_grid[parameter_name]
    assert first_search.cv.n_splits == configuration["folds"]
    assert first_search.cv.random_state == configuration["seed"]
    assert first_search.cv.shuffle is True
    assert first_search.scoring == {"f1_macro": "f1_macro", "accuracy": "accuracy"}
    assert first_search.refit == "f1_macro" and first_search.n_jobs == -1
    assert first_search.verbose == 1 and first_search.return_train_score is True


def test_build_cv_predictions_returns_one_known_label_per_row():
    training_data = create_synthetic_training_data()
    configuration = load_config("configs/default.yaml")
    pipeline = build_decision_tree_pipeline(None, configuration)
    label_ids = training_data["label_ids"]
    predictions = build_cv_predictions(
        pipeline, training_data["features"], label_ids, training_data["image_ids"], configuration,
    )
    assert predictions.shape == label_ids.shape
    assert set(predictions).issubset(set(label_ids))


def test_save_and_load_model_preserves_predictions(tmp_path):
    training_data = create_synthetic_training_data()
    features = training_data["features"]
    pipeline = build_svm_pipeline(None)
    pipeline.fit(features, training_data["label_ids"])
    model_path = tmp_path / "models" / "model.joblib"
    save_model(pipeline, model_path)
    restored_model = load_model(model_path)
    original_predictions = pipeline.predict(features)
    restored_predictions = restored_model.predict(features)
    np.testing.assert_array_equal(restored_predictions, original_predictions)
    with pytest.raises(ValueError, match="Model file does not exist"):
        load_model(tmp_path / "missing.joblib")


def test_select_best_candidate_breaks_ties_by_std_then_order():
    candidates = [
        {"model_name": "svm", "best_cv_macro_f1": 0.80, "cv_std": 0.01},
        {"model_name": "svm", "best_cv_macro_f1": 0.90, "cv_std": 0.06},
        {"model_name": "decision_tree", "best_cv_macro_f1": 0.90, "cv_std": 0.02},
        {"model_name": "decision_tree", "best_cv_macro_f1": 0.90, "cv_std": 0.02},
    ]
    assert train.select_best_candidate(candidates[:2]) is candidates[1]
    assert train.select_best_candidate(candidates) is candidates[2]


def create_summary_candidates():
    candidates = []
    for model_name, class_weight in train.CANDIDATE_OPTIONS:
        candidates.append({
            "model_name": model_name, "class_weight": class_weight, "best_params": {"depth": None},
            "train_accuracy": 0.93456, "cv_accuracy": 0.82346, "cv_accuracy_std": 0.02567, "train_macro_f1": 0.92344,
            "best_cv_macro_f1": 0.81234, "cv_std": 0.04321, "fit_seconds": 1.23456,
        })
    return candidates


def test_write_cv_summary_writes_four_rows_with_expected_columns(tmp_path):
    candidates = create_summary_candidates()
    model_directory = tmp_path / "models"
    train.write_cv_summary(candidates, model_directory)
    with (model_directory / "cv_results.csv").open(newline="", encoding="utf-8") as report_file:
        reader = csv.DictReader(report_file)
        assert tuple(reader.fieldnames) == (
            "model_name", "class_weight", "best_params", "train_accuracy", "cv_accuracy", "cv_accuracy_std",
            "train_macro_f1", "best_cv_macro_f1", "cv_std", "fit_seconds",
        )
        summary_rows = list(reader)
    assert len(summary_rows) == 4
    assert [row["class_weight"] for row in summary_rows] == ["none", "balanced", "none", "balanced"]
    assert json.loads(summary_rows[0]["best_params"]) == {"depth": None}
    first_values = [summary_rows[0][key] for key in train.CV_METRIC_COLUMNS]
    assert first_values == ["0.9346", "0.8235", "0.0257", "0.9234", "0.8123", "0.0432", "1.2346"]


def test_print_training_summary_has_four_data_rows_and_formatted_gap(capsys):
    train.print_training_summary(create_summary_candidates())
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 5
    assert lines[0].split() == [
        "model", "class_weight", "train_acc", "val_acc(train_cv_mean)", "train_f1", "val_f1(train_cv_mean)", "gap_f1",
    ]
    assert lines[1].split() == ["svm", "none", "0.935", "0.823", "0.923", "0.812", "0.111"]
    assert lines[-1].split()[:2] == ["decision_tree", "balanced"]


def test_summarize_candidate_reads_metrics_at_best_index():
    result_values = {
        "mean_train_accuracy": [0.1, 0.95], "mean_test_accuracy": [0.2, 0.85], "std_test_accuracy": [0.3, 0.05],
        "mean_train_f1_macro": [0.4, 0.94], "std_test_f1_macro": [0.5, 0.04],
    }
    search = SimpleNamespace(
        best_index_=1, best_score_=0.84, best_estimator_=object(), best_params_={"svc__C": 1}, cv_results_=result_values,
    )
    summary = train.summarize_candidate("svm", None, search, 1.5)
    assert set(summary) == set(train.CV_SUMMARY_COLUMNS) | {"best_estimator"}
    metric_values = [summary[key] for key in train.CV_METRIC_COLUMNS]
    assert np.isfinite(metric_values).all()
    assert metric_values == [0.95, 0.85, 0.05, 0.94, 0.84, 0.04, 1.5]
    result_values["mean_test_accuracy"][1] = np.inf
    with pytest.raises(ValueError, match="nonfinite training metrics"):
        train.summarize_candidate("svm", None, search, 1.5)


def create_training_archive_arrays():
    feature_names = [f"feature_{column}" for column in range(403)]
    return {
        "features": np.zeros((3, 403), dtype=np.float64),
        "feature_names": np.array(feature_names),
        "label_ids": np.array([0, 1, 2]),
        "image_ids": np.array([12, 24, 36]),
        "label_names": np.array(["lime", "amber", "green"]),
    }


def test_load_training_data_reads_synthetic_train_rows(tmp_path):
    training_arrays = create_training_archive_arrays()
    archive_path = tmp_path / "features_train.npz"
    np.savez_compressed(archive_path, **training_arrays)
    training_data = train.load_training_data(archive_path)
    for array_name, expected_values in training_arrays.items():
        np.testing.assert_array_equal(training_data[array_name], expected_values)


@pytest.mark.parametrize("nonfinite_value", [np.nan, np.inf])
def test_load_training_data_rejects_nonfinite_features(tmp_path, nonfinite_value):
    training_arrays = create_training_archive_arrays()
    training_arrays["features"][0, 0] = nonfinite_value
    archive_path = tmp_path / "features_train.npz"
    np.savez_compressed(archive_path, **training_arrays)
    with pytest.raises(ValueError, match="finite real numbers"):
        train.load_training_data(archive_path)


@pytest.mark.parametrize("array_name, error_message", [
    ("image_ids", "one value per feature row"), ("label_ids", "one value per feature row"),
    ("feature_names", "one name per feature column"),
])
def test_load_training_data_rejects_incompatible_shapes(tmp_path, array_name, error_message):
    training_arrays = create_training_archive_arrays()
    training_arrays[array_name] = training_arrays[array_name][:-1]
    archive_path = tmp_path / "features_train.npz"
    np.savez_compressed(archive_path, **training_arrays)
    with pytest.raises(ValueError, match=error_message):
        train.load_training_data(archive_path)


def test_load_training_data_rejects_missing_keys(tmp_path):
    training_arrays = create_training_archive_arrays()
    del training_arrays["image_ids"]
    archive_path = tmp_path / "features_train.npz"
    np.savez_compressed(archive_path, **training_arrays)
    with pytest.raises(ValueError, match="missing keys.*image_ids"):
        train.load_training_data(archive_path)


def test_validate_label_names_requires_order_by_label_id():
    label_by_code = {2: "green", 0: "lime", 1: "amber"}
    train.validate_label_names(np.array(["lime", "amber", "green"]), label_by_code)
    with pytest.raises(ValueError, match="label ID order"):
        train.validate_label_names(np.array(["amber", "lime", "green"]), label_by_code)


def check_saved_model_info(model_directory, best_candidate, training_data):
    model_info = json.loads((model_directory / "best_model_info.json").read_text(encoding="utf-8"))
    assert set(model_info) == {
        "model_name", "class_weight", "best_params", "best_cv_macro_f1", "cv_std",
        "train_macro_f1", "train_accuracy", "cv_accuracy",
        "feature_count", "train_rows", "class_names", "sklearn_version",
    }
    saved_model = load_model(model_directory / "best_model.joblib")
    assert model_info["class_names"] == training_data["label_names"][saved_model.classes_].tolist()
    assert model_info["model_name"] == best_candidate["model_name"]
    assert model_info["feature_count"] == 6 and model_info["train_rows"] == 60
    for score_name in ("train_macro_f1", "train_accuracy", "cv_accuracy"):
        assert model_info[score_name] == best_candidate[score_name]


def check_saved_cv_diagnostics(model_directory, training_data):
    class_names = training_data["label_names"].tolist()
    matrix_table = pd.read_csv(model_directory / "cv_confusion_matrix.csv", index_col=0)
    per_class_table = pd.read_csv(model_directory / "cv_per_class.csv")
    assert matrix_table.shape == (3, 3)
    assert matrix_table.index.tolist() == matrix_table.columns.tolist() == class_names
    assert matrix_table.to_numpy().sum() == 60
    assert per_class_table.columns.tolist() == ["class_name", "precision", "recall", "f1", "support"]
    assert per_class_table.shape == (3, 5)
    assert per_class_table["class_name"].tolist() == class_names
    assert per_class_table["support"].tolist() == [20, 20, 20]


def test_train_select_and_save_model_on_synthetic_features(tmp_path, capsys):
    training_data = create_synthetic_training_data()
    configuration = load_config("configs/default.yaml")
    parameter_grids = create_small_parameter_grids()
    candidates = train.train_candidate_models(training_data, parameter_grids, configuration)
    candidate_order = [(candidate["model_name"], candidate["class_weight"]) for candidate in candidates]
    assert candidate_order == list(train.CANDIDATE_OPTIONS)
    assert capsys.readouterr().out.count("fit_seconds=") == 4
    best_candidate = train.select_best_candidate(candidates)
    model_directory = tmp_path / "models" / "artifacts"
    train.write_cv_summary(candidates, model_directory)
    train.write_cv_diagnostics(best_candidate, training_data, configuration, model_directory)
    diagnostics_output = capsys.readouterr().out
    assert "Train CV out-of-fold accuracy=" in diagnostics_output
    assert "actual / predicted" in diagnostics_output
    train.save_best_model(best_candidate, model_directory, training_data)
    assert {path.name for path in model_directory.iterdir()} == {
        "cv_results.csv", "cv_confusion_matrix.csv", "cv_per_class.csv", "best_model.joblib", "best_model_info.json",
    }
    check_saved_cv_diagnostics(model_directory, training_data)
    check_saved_model_info(model_directory, best_candidate, training_data)
