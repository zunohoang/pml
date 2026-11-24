import json

import numpy as np
import pandas as pd
import pytest

from coffee_leaf.analysis import summaries


@pytest.fixture
def label_table():
    return pd.DataFrame({
        "predominant_stress": [0, 0, 1, 1, 5], "has_image": [True, False, True, True, True],
        "miner": [0, 0, 1, 1, 0], "rust": [0, 0, 0, 1, 1],
        "phoma": [0, 0, 0, 0, 1], "cercospora": [0, 0, 0, 0, 0],
        "severity": [0, 0, 1, 4, 9],
    })


@pytest.fixture
def label_by_code():
    return {0: "healthy", 1: "miner"}


def test_summarize_dataset_and_split_by_class(label_table, label_by_code):
    overview = summaries.summarize_dataset(label_table, label_by_code)
    assert overview["rows"].tolist() == [2, 2, 1]
    assert overview["images_with_file"].tolist() == [1, 2, 1]
    assert overview["used"].tolist() == [True, True, False]
    split_counts = summaries.summarize_split(overview, ["healthy", "miner"])
    assert split_counts.to_dict("records") == [
        {"label": "healthy", "total": 1, "train": 1, "test": 0},
        {"label": "miner", "total": 2, "train": 1, "test": 1},
    ]
    with pytest.raises(ValueError, match="Train count"):
        summaries.summarize_split(overview, ["healthy", "healthy"])
    with pytest.raises(ValueError, match="boolean"):
        summaries.summarize_dataset(label_table.assign(has_image=1), label_by_code)


def test_summarize_multilabel_flags_in_column_order(label_table, label_by_code):
    crosstab, counts = summaries.summarize_multilabel(label_table, label_by_code)
    assert crosstab.loc[0, "0000"] == 2
    assert crosstab.loc[1, "1000"] == crosstab.loc[1, "1100"] == crosstab.loc[5, "0110"] == 1
    assert counts == {"multilabel_rows": 2, "labeled_multilabel_rows": 1}
    invalid = label_table.assign(rust=9)
    with pytest.raises(ValueError, match="only 0 and 1"):
        summaries.summarize_multilabel(invalid, label_by_code)


def test_summarize_severity_uses_actual_minimum_and_maximum(label_table, label_by_code):
    severity = summaries.summarize_severity(label_table, label_by_code).set_index("label")
    assert severity.loc["miner", ["n", "minimum", "median", "maximum"]].tolist() == [2, 1, 2.5, 4]
    assert severity.loc["healthy", "maximum"] == 0
    with pytest.raises(ValueError, match="finite"):
        summaries.summarize_severity(label_table.assign(severity=np.nan), label_by_code)


def test_summarize_image_sizes_counts_pairs():
    sizes = summaries.summarize_image_sizes([(10, 20), (9, 5), (10, 20)])
    counts = sizes.set_index(["width", "height"])["image_count"].to_dict()
    assert counts == {(9, 5): 1, (10, 20): 2}
    with pytest.raises(ValueError, match="positive integers"):
        summaries.summarize_image_sizes([(0, 20)])


def test_lesion_ratios_and_zero_blocks_use_different_rules():
    features = np.array([[0.0, 0, 1], [0, 2, 1], [0.25, 1, 1], [0.75, 1, 1]])
    names = ["lesion_shape_area_ratio", "lesion_lab_mean_l", "leaf_value"]
    regions = ["lesion", "lesion", "leaf"]
    labels = ["healthy", "healthy", "miner", "miner"]
    ratios = summaries.extract_lesion_ratios(features, names)
    np.testing.assert_array_equal(ratios, [0, 0, 0.25, 0.75])
    stats = summaries.summarize_lesion_ratios(ratios, labels).set_index("label")
    statistics = stats.loc["miner", ["n", "mean", "median", "q1", "q3"]].tolist()
    assert statistics == [2, 0.5, 0.5, 0.375, 0.625]
    blocks = summaries.summarize_zero_lesion_blocks(features, regions, labels)
    assert blocks.set_index("label")["zero_lesion_block"].to_dict() == {"healthy": 1, "miner": 0}


def test_lesion_validators_reject_invalid_inputs():
    with pytest.raises(ValueError, match="exactly one"):
        summaries.extract_lesion_ratios([[0, 1]], ["other", "columns"])
    with pytest.raises(ValueError, match="between 0 and 1"):
        summaries.summarize_lesion_ratios([-1], ["healthy"])
    with pytest.raises(ValueError, match="lesion columns"):
        summaries.summarize_zero_lesion_blocks([[0, 1]], ["leaf", "leaf"], ["healthy"])


def test_constant_columns_include_nonzero_constant_and_metadata():
    features = [[0.1, 0, 1], [0.1, 0, 2], [0.1, 0, 3]]
    groups = ["color", "shape", "color"]
    regions = ["leaf", "lesion", "leaf"]
    constants = summaries.summarize_constant_columns(features, ["a", "b", "c"], groups, regions)
    assert constants["feature_name"].tolist() == ["a", "b"]
    assert constants["feature_region"].tolist() == ["leaf", "lesion"]
    assert constants["value"].tolist() == [0.1, 0]
    with pytest.raises(ValueError, match="one class label"):
        summaries.summarize_constant_columns(features, ["short"], groups, regions)


def test_correlation_counts_unordered_pairs_and_excludes_constants():
    features = [[1, -1, 1, 0], [2, -2, -1, 0], [3, -3, -1, 0], [4, -4, 1, 0]]
    correlation, metadata = summaries.build_correlation_matrix(
        features, ["a", "b", "c", "d"], ["texture", "color", "gradient", "shape"], ["leaf"] * 4,
    )
    assert metadata["feature_name"].tolist() == ["b", "c", "a"]
    stats, top_pairs = summaries.summarize_correlation_pairs(correlation)
    assert stats.loc[0, ["pairs", "high_pairs", "linked_columns"]].tolist() == [3, 1, 2]
    assert stats.loc[0, "linked_share"] == pytest.approx(2 / 3)
    assert top_pairs.loc[0, "r"] == pytest.approx(-1)
    with pytest.raises(ValueError, match="at least two"):
        summaries.build_correlation_matrix([[0, 1], [0, 2]], ["a", "b"], ["color"] * 2, ["leaf"] * 2)
    with pytest.raises(ValueError, match="row and column names"):
        summaries.summarize_correlation_pairs(correlation.iloc[::-1])


def test_compute_pca_retains_two_components_with_valid_variance():
    features = [[1, -1, 0], [2, -2, 0], [3, -3, 0], [4, -4, 0]]
    scores, variance, components = summaries.compute_pca(features, 42)
    assert scores.shape == (4, 2) and len(variance) == 3
    assert variance["explained_variance_ratio"].sum() <= 1 + 1e-12
    assert components == 1
    with pytest.raises(ValueError, match="nonconstant"):
        summaries.compute_pca([[1, 1], [1, 1]], 42)
    with pytest.raises(ValueError, match="finite"):
        summaries.compute_pca([[1, np.inf], [1, 1]], 42)


def test_write_summary_json_preserves_keys(tmp_path):
    pytest.importorskip("matplotlib")
    from scripts import run_eda

    summary = {key: 0 for key in run_eda.SUMMARY_KEYS}
    summary["classes"] = {"healthy": {"total": 2, "train": 1, "test": 1}}
    summary["zero_lesion_block_by_class"] = {"healthy": 1}
    run_eda.write_summary_json(summary, tmp_path)
    loaded = json.loads((tmp_path / "eda_summary.json").read_text(encoding="utf-8"))
    assert loaded == summary
    with pytest.raises(ValueError, match="missing keys"):
        run_eda.write_summary_json({}, tmp_path)


def build_plot_arguments(name):
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    mask = np.ones((4, 5), dtype=bool)
    correlation = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]])
    metadata = pd.DataFrame({"feature_group": ["color", "texture"], "feature_region": ["leaf", "lesion"]})
    arguments = {
        "plot_class_distribution": (pd.DataFrame({"label": ["healthy"], "total": [2], "train": [1], "test": [1]}),),
        "plot_multilabel": (pd.DataFrame({"0000": [2], "1100": [1]}),),
        "plot_severity_by_class": (pd.DataFrame({"label": ["healthy", "rust"], "severity": [0, 2]}),),
        "plot_sample_images": ({"healthy": [(1, image)]},),
        "plot_segmentation_overlays": ({"healthy": [(1, image, mask, ~mask)]},),
        "plot_lesion_ratio_by_class": (pd.DataFrame({"label": ["healthy", "rust"], "lesion_ratio": [0.1, 0.2]}),),
        "plot_feature_correlation": (correlation, metadata),
        "plot_pca_scatter": (np.zeros((2, 2)), np.array(["healthy", "rust"]),
                             pd.DataFrame({"explained_variance_ratio": [0.3, 0.2]})),
    }
    return arguments[name]


@pytest.mark.parametrize("name", [
    "plot_class_distribution", "plot_multilabel", "plot_severity_by_class", "plot_sample_images",
    "plot_segmentation_overlays", "plot_lesion_ratio_by_class", "plot_feature_correlation", "plot_pca_scatter",
])
def test_plots_return_figures_that_save_to_png(name, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from coffee_leaf.analysis import plots

    draw_figure = getattr(plots, name)
    arguments = build_plot_arguments(name)
    figure = draw_figure(*arguments)
    output_path = tmp_path / f"{name}.png"
    try:
        assert isinstance(figure, matplotlib.figure.Figure)
        figure.savefig(output_path, dpi=120)
        assert output_path.read_bytes().startswith(b"\x89PNG")
    finally:
        plots.plt.close(figure)
