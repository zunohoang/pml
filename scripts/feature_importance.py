from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from coffee_leaf.config import load_config
from coffee_leaf.evaluation.feature_importance import (
    build_feature_table,
    extract_tree_importances,
    select_best_tree_name,
    summarize_by_column,
    summarize_by_group_and_region,
)
from coffee_leaf.models.persistence import load_model


CONFIG_PATH = Path("configs/default.yaml")
TRAIN_FEATURES_PATH = Path("data/processed/features_train.npz")
FEATURE_METADATA_KEYS = ("feature_names", "feature_groups", "feature_regions")
EXPECTED_FEATURE_COUNT = 403

IMPORTANCE_FILENAMES = (
    ("features", "tree_feature_importance.csv"),
    ("by_group", "tree_importance_by_group.csv"),
    ("by_region", "tree_importance_by_region.csv"),
    ("by_group_region", "tree_importance_by_group_region.csv"),
)


def load_feature_metadata(archive_path):
    with np.load(archive_path, allow_pickle=False) as archive:
        missing_keys = set(FEATURE_METADATA_KEYS) - set(archive.files)
        if missing_keys:
            raise ValueError(f"Training archive is missing feature metadata: {sorted(missing_keys)}.")
        metadata = {}
        for key in FEATURE_METADATA_KEYS:
            values = archive[key]
            if values.shape != (EXPECTED_FEATURE_COUNT,) or values.dtype.kind != "U":
                raise ValueError(f"{key} must contain {EXPECTED_FEATURE_COUNT} strings.")
            metadata[key] = values
    return metadata


def load_cv_results(model_dir):
    cv_results_path = Path(model_dir) / "cv_results.csv"
    return pd.read_csv(cv_results_path, keep_default_na=False)


def load_best_tree_pipeline(cv_results, model_dir):
    candidate_name = select_best_tree_name(cv_results)
    model_path = Path(model_dir) / "candidates" / f"{candidate_name}.joblib"
    pipeline = load_model(model_path)
    return candidate_name, pipeline


def compute_importance_tables(pipeline, feature_metadata):
    feature_names = feature_metadata["feature_names"]
    feature_groups = feature_metadata["feature_groups"]
    feature_regions = feature_metadata["feature_regions"]
    expected_count = len(feature_names)
    importances = extract_tree_importances(pipeline, expected_count)
    feature_table = build_feature_table(feature_names, feature_groups, feature_regions, importances)
    return {
        "features": feature_table,
        "by_group": summarize_by_column(feature_table, "group"),
        "by_region": summarize_by_column(feature_table, "region"),
        "by_group_region": summarize_by_group_and_region(feature_table),
    }


def write_importance_files(importance_tables, model_dir):
    model_directory = Path(model_dir)
    model_directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=model_directory) as temporary_directory:
        staged_directory = Path(temporary_directory)
        for table_name, filename in IMPORTANCE_FILENAMES:
            staged_path = staged_directory / filename
            importance_tables[table_name].to_csv(staged_path, index=False, float_format="%.4f")
        for table_name, filename in IMPORTANCE_FILENAMES:
            staged_path = staged_directory / filename
            staged_path.replace(model_directory / filename)


def prepare_importance_summary(importance_tables):
    top_features = importance_tables["features"].head(10)
    feature_rows = list(zip(top_features["feature_name"], top_features["importance"]))
    group_rows = list(importance_tables["by_group"].itertuples(index=False, name=None))
    region_rows = list(importance_tables["by_region"].itertuples(index=False, name=None))
    return feature_rows, group_rows, region_rows


def print_importance_summary(candidate_name, summary_rows):
    feature_rows, group_rows, region_rows = summary_rows
    print(f"Decision Tree selected by CV: {candidate_name}")
    print(f"{'Top 10 features':<34} {'importance':>10}")
    for feature_name, importance in feature_rows:
        print(f"{feature_name:<34} {importance:>10.3f}")
    for heading, rows in (("Group", group_rows), ("Region", region_rows)):
        print(f"{heading:<16} {'importance':>10} {'feature_count':>13}")
        for name, importance, feature_count in rows:
            print(f"{name:<16} {importance:>10.3f} {feature_count:>13}")


def main():
    configuration = load_config(CONFIG_PATH)
    model_dir = configuration["model_dir"]
    feature_metadata = load_feature_metadata(TRAIN_FEATURES_PATH)
    cv_results = load_cv_results(model_dir)
    candidate_name, pipeline = load_best_tree_pipeline(cv_results, model_dir)
    importance_tables = compute_importance_tables(pipeline, feature_metadata)
    write_importance_files(importance_tables, model_dir)
    summary_rows = prepare_importance_summary(importance_tables)
    print_importance_summary(candidate_name, summary_rows)


if __name__ == "__main__":
    main()
