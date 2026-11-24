import numpy as np
import pandas as pd
import pytest


matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from coffee_leaf.evaluation.plots import (
    plot_confusion_matrices,
    plot_per_class_scores,
    plot_tree_importance,
    plot_validation_scores,
)


def check_figure_can_be_saved(figure, tmp_path):
    assert isinstance(figure, matplotlib.figure.Figure)
    output_path = tmp_path / "plot.png"
    try:
        figure.savefig(output_path)
        assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        matplotlib.pyplot.close(figure)


def test_plot_validation_scores_shows_cv_error_bars(tmp_path):
    cv_results = pd.DataFrame({
        "model_name": ["svm", "decision_tree"],
        "class_weight": ["none", "balanced"],
        "train_accuracy": [0.95, 0.92],
        "cv_accuracy": [0.80, 0.70],
        "cv_accuracy_std": [0.02, 0.03],
        "train_macro_f1": [0.94, 0.91],
        "best_cv_macro_f1": [0.79, 0.68],
        "cv_std": [0.03, 0.04],
    })
    figure = plot_validation_scores(cv_results)
    check_figure_can_be_saved(figure, tmp_path)
    assert [axis.get_title() for axis in figure.axes] == ["Accuracy", "Macro-F1"]
    for axis in figure.axes:
        assert [tick.get_text() for tick in axis.get_xticklabels()] == ["svm\nnone", "decision_tree\nbalanced"]
        assert any(container.get_label() == "Validation (CV)" and container.errorbar for container in axis.containers)


def test_plot_confusion_matrices_preserves_counts_and_normalizes_rows(tmp_path):
    class_names = ["zeta", "alpha"]
    confusion_table = pd.DataFrame([[3, 1], [0, 0]], index=class_names, columns=class_names)
    figure = plot_confusion_matrices(confusion_table)
    check_figure_can_be_saved(figure, tmp_path)
    counts_axis, rates_axis = figure.axes
    np.testing.assert_array_equal(counts_axis.images[0].get_array(), [[3, 1], [0, 0]])
    np.testing.assert_array_equal(rates_axis.images[0].get_array(), [[0.75, 0.25], [0, 0]])
    assert rates_axis.xaxis.label.get_text() == "Predicted"
    assert [tick.get_text() for tick in rates_axis.get_yticklabels()] == class_names


def test_plot_per_class_scores_shows_three_metrics(tmp_path):
    per_class_table = pd.DataFrame({
        "class_name": ["zeta", "alpha"],
        "precision": [0.9, 0.7],
        "recall": [0.8, 0.6],
        "f1": [0.85, 0.65],
        "support": [12, 8],
    })
    figure = plot_per_class_scores(per_class_table)
    check_figure_can_be_saved(figure, tmp_path)
    axis = figure.axes[0]
    assert len(figure.axes) == 1
    assert [container.get_label() for container in axis.containers] == ["precision", "recall", "f1"]
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["zeta", "alpha"]


def test_plot_tree_importance_shows_three_horizontal_panels(tmp_path):
    feature_names = [f"feature_{position}" for position in range(20)]
    feature_table = pd.DataFrame({"feature_name": feature_names, "importance": np.linspace(0.20, 0.01, 20)})
    group_table = pd.DataFrame({"group": ["color", "shape"], "importance": [0.75, 0.25]})
    region_table = pd.DataFrame({"region": ["leaf", "lesion"], "importance": [0.80, 0.20]})
    figure = plot_tree_importance(feature_table, group_table, region_table)
    check_figure_can_be_saved(figure, tmp_path)
    assert [axis.get_title() for axis in figure.axes] == ["Top 15 features", "By group", "By region"]
    assert [len(axis.patches) for axis in figure.axes] == [15, 2, 2]
    assert [tick.get_text() for tick in figure.axes[0].get_yticklabels()] == feature_names[:15]
    for axis in figure.axes:
        assert axis.yaxis_inverted()
        assert axis.xaxis.label.get_text() == "Importance"
