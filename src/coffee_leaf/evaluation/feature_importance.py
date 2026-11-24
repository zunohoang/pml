import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


IMPORTANCE_SUM_TOLERANCE = 1e-6
TREE_SCORE_COLUMNS = ("best_cv_macro_f1", "cv_std")


def select_best_tree_name(cv_results):
    required_columns = {"model_name", "class_weight", *TREE_SCORE_COLUMNS}
    if not required_columns.issubset(cv_results.columns):
        raise ValueError("CV results are missing columns required to select a tree.")
    is_tree_mask = cv_results["model_name"] == "decision_tree"
    tree_results = cv_results.loc[is_tree_mask].copy()
    if tree_results.empty:
        raise ValueError("CV results do not contain a Decision Tree candidate.")
    weight_ranks = tree_results["class_weight"].map({"none": 0, "balanced": 1})
    if weight_ranks.isna().any():
        raise ValueError("Decision Tree class_weight must be 'none' or 'balanced'.")
    score_columns = list(TREE_SCORE_COLUMNS)
    tree_scores = tree_results[score_columns].astype(float)
    if not np.isfinite(tree_scores.to_numpy()).all():
        raise ValueError("Decision Tree CV scores must be finite.")
    tree_results[score_columns] = tree_scores
    tree_results["class_weight_rank"] = weight_ranks
    rank_columns = [*score_columns, "class_weight_rank"]
    ranked_trees = tree_results.sort_values(rank_columns, ascending=[False, True, True])
    class_weight = ranked_trees.iloc[0]["class_weight"]
    return f"decision_tree_{class_weight}"


def find_tree_classifier(pipeline):
    for step_name, estimator in getattr(pipeline, "steps", ()):
        if isinstance(estimator, DecisionTreeClassifier):
            return estimator
    raise ValueError("The pipeline must contain a DecisionTreeClassifier step.")


def extract_tree_importances(pipeline, expected_count):
    tree_classifier = find_tree_classifier(pipeline)
    importances = getattr(tree_classifier, "feature_importances_", None)
    if importances is None:
        raise ValueError("The Decision Tree must be fitted before extracting importances.")
    if importances.shape != (expected_count,):
        raise ValueError(f"The Decision Tree must have {expected_count} feature importances.")
    if not np.isfinite(importances).all() or (importances < 0).any():
        raise ValueError("Feature importances must be finite and nonnegative.")
    importance_total = importances.sum()
    if abs(importance_total - 1.0) > IMPORTANCE_SUM_TOLERANCE:
        raise ValueError("Feature importances must sum to 1 within 1e-6.")
    return importances.copy()


def build_feature_table(feature_names, feature_groups, feature_regions, importances):
    if not len(feature_names) == len(feature_groups) == len(feature_regions) == len(importances):
        raise ValueError("Feature names, groups, regions, and importances must have equal lengths.")
    feature_table = pd.DataFrame({
        "feature_name": feature_names,
        "group": feature_groups,
        "region": feature_regions,
        "importance": importances,
    })
    return feature_table.sort_values("importance", ascending=False, kind="stable", ignore_index=True)


def summarize_importances(feature_table, column_names):
    grouped_features = feature_table.groupby(column_names, as_index=False)
    summary = grouped_features.agg(importance=("importance", "sum"), feature_count=("importance", "size"))
    return summary.sort_values("importance", ascending=False, kind="stable", ignore_index=True)


def summarize_by_column(feature_table, column_name):
    return summarize_importances(feature_table, [column_name])


def summarize_by_group_and_region(feature_table):
    return summarize_importances(feature_table, ["group", "region"])
