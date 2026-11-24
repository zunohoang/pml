import csv

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from coffee_leaf.evaluation.feature_importance import (
    build_feature_table,
    extract_tree_importances,
    select_best_tree_name,
    summarize_by_column,
    summarize_by_group_and_region,
)
from coffee_leaf.models.persistence import save_model
from scripts import feature_importance


def build_cv_results(none_score, none_std, balanced_score, balanced_std):
    candidates = [
        ("svm", "none", 0.99, 0.01),
        ("decision_tree", "balanced", balanced_score, balanced_std),
        ("decision_tree", "none", none_score, none_std),
    ]
    columns = ["model_name", "class_weight", "best_cv_macro_f1", "cv_std"]
    return pd.DataFrame(candidates, columns=columns)


@pytest.mark.parametrize(("none_score", "none_std", "balanced_score", "balanced_std", "expected_name"), [
    (0.80, 0.10, 0.78, 0.01, "decision_tree_none"),
    (0.80, 0.10, 0.80, 0.02, "decision_tree_balanced"),
    (0.80, 0.02, 0.80, 0.02, "decision_tree_none"),
])
def test_select_best_tree_name_ranks_tree_rows(none_score, none_std, balanced_score, balanced_std, expected_name):
    cv_results = build_cv_results(none_score, none_std, balanced_score, balanced_std)
    assert select_best_tree_name(cv_results) == expected_name


def test_select_best_tree_name_rejects_missing_tree():
    cv_results = build_cv_results(0.80, 0.10, 0.78, 0.02)
    with pytest.raises(ValueError, match="do not contain a Decision Tree"):
        select_best_tree_name(cv_results.iloc[:1])


def fit_tree_pipeline(feature_count):
    random_source = np.random.default_rng(42)
    features = random_source.normal(size=(36, feature_count))
    first_feature = features[:, 0]
    label_ids = (first_feature > 0).astype(np.int64)
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(max_depth=2, random_state=42)),
    ])
    pipeline.fit(features, label_ids)
    return pipeline


def test_extract_tree_importances_returns_copy_with_unit_sum():
    pipeline = fit_tree_pipeline(5)
    importances = extract_tree_importances(pipeline, 5)
    assert importances.shape == (5,)
    assert importances.sum() == pytest.approx(1.0)
    importances.fill(0)
    assert extract_tree_importances(pipeline, 5).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="6 feature importances"):
        extract_tree_importances(pipeline, 6)


def test_extract_tree_importances_rejects_pipeline_without_tree():
    pipeline = Pipeline([("scale", StandardScaler())])
    with pytest.raises(ValueError, match="DecisionTreeClassifier step"):
        extract_tree_importances(pipeline, 5)


def test_extract_tree_importances_rejects_tree_without_splits():
    pipeline = Pipeline([("tree", DecisionTreeClassifier(random_state=42))])
    pipeline.fit(np.zeros((4, 5)), np.arange(4))
    with pytest.raises(ValueError, match="sum to 1"):
        extract_tree_importances(pipeline, 5)


@pytest.fixture
def feature_table():
    return build_feature_table(
        ["first", "second", "third", "fourth"],
        ["color", "texture", "color", "texture"],
        ["leaf", "leaf", "lesion", "leaf"],
        [0.15, 0.40, 0.40, 0.05],
    )


def test_build_feature_table_sorts_descending_and_preserves_ties(feature_table):
    assert feature_table.columns.tolist() == ["feature_name", "group", "region", "importance"]
    assert feature_table["feature_name"].tolist() == ["second", "third", "first", "fourth"]
    with pytest.raises(ValueError, match="equal lengths"):
        build_feature_table(["first"], ["color"], ["leaf"], [1.0, 0.0])


@pytest.mark.parametrize(("column_name", "first_name", "first_total", "first_count"), [
    ("group", "color", 0.55, 2),
    ("region", "leaf", 0.60, 3),
])
def test_summarize_by_column_preserves_importance_and_counts(feature_table, column_name, first_name, first_total, first_count):
    summary = summarize_by_column(feature_table, column_name)
    assert summary.columns.tolist() == [column_name, "importance", "feature_count"]
    assert summary["importance"].sum() == pytest.approx(1.0)
    assert summary["feature_count"].sum() == len(feature_table)
    assert summary.iloc[0][column_name] == first_name
    assert summary.iloc[0]["importance"] == pytest.approx(first_total)
    assert summary.iloc[0]["feature_count"] == first_count


def test_summarize_by_group_and_region_preserves_importance_and_counts(feature_table):
    summary = summarize_by_group_and_region(feature_table)
    assert summary.columns.tolist() == ["group", "region", "importance", "feature_count"]
    assert summary["importance"].sum() == pytest.approx(1.0)
    assert summary["feature_count"].sum() == len(feature_table)
    assert tuple(summary.iloc[0]) == ("texture", "leaf", pytest.approx(0.45), 2)


def create_synthetic_metadata():
    return {
        "feature_names": np.array([f"feature_{position}" for position in range(403)]),
        "feature_groups": np.resize(["color", "texture", "gradient", "shape"], 403),
        "feature_regions": np.resize(["leaf", "lesion", "leaf"], 403),
    }


def test_load_feature_metadata_only_reads_requested_arrays(tmp_path):
    metadata = create_synthetic_metadata()
    archive_path = tmp_path / "synthetic.npz"
    unreadable_features = np.array([None], dtype=object)
    np.savez_compressed(archive_path, features=unreadable_features, **metadata)
    loaded_metadata = feature_importance.load_feature_metadata(archive_path)
    assert set(loaded_metadata) == set(metadata)
    for key in metadata:
        np.testing.assert_array_equal(loaded_metadata[key], metadata[key])


def test_load_best_tree_pipeline_uses_cv_name(tmp_path):
    cv_results = build_cv_results(0.80, 0.02, 0.80, 0.01)
    cv_results.to_csv(tmp_path / "cv_results.csv", index=False)
    pipeline = fit_tree_pipeline(5)
    save_model(pipeline, tmp_path / "candidates" / "decision_tree_balanced.joblib")
    saved_cv_results = feature_importance.load_cv_results(tmp_path)
    candidate_name, loaded_pipeline = feature_importance.load_best_tree_pipeline(saved_cv_results, tmp_path)
    assert candidate_name == "decision_tree_balanced"
    np.testing.assert_array_equal(extract_tree_importances(loaded_pipeline, 5), extract_tree_importances(pipeline, 5))


def test_write_importance_files_creates_four_csv_reports(tmp_path):
    pipeline = fit_tree_pipeline(403)
    metadata = create_synthetic_metadata()
    importance_tables = feature_importance.compute_importance_tables(pipeline, metadata)
    model_dir = tmp_path / "artifacts"
    feature_importance.write_importance_files(importance_tables, model_dir)
    expected_columns = {
        "tree_feature_importance.csv": ["feature_name", "group", "region", "importance"],
        "tree_importance_by_group.csv": ["group", "importance", "feature_count"],
        "tree_importance_by_region.csv": ["region", "importance", "feature_count"],
        "tree_importance_by_group_region.csv": ["group", "region", "importance", "feature_count"],
    }
    assert {path.name for path in model_dir.iterdir()} == set(expected_columns)
    for filename, columns in expected_columns.items():
        with (model_dir / filename).open(newline="", encoding="utf-8") as report:
            reader = csv.DictReader(report)
            rows = list(reader)
        assert reader.fieldnames == columns
        assert all(len(row["importance"].split(".")[-1]) == 4 for row in rows)
        if filename == "tree_feature_importance.csv":
            assert len(rows) == 403
