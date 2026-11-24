import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits

from coffee_leaf.analysis import plots, summaries
from coffee_leaf.config import load_config
from coffee_leaf.data.loader import read_dataset_table
from coffee_leaf.features.extractor import EXPECTED_FEATURE_COUNT
from coffee_leaf.preprocessing.image import find_glare_mask, load_resized_rgb, preprocess_rgb
from coffee_leaf.preprocessing.segmentation import find_leaf_mask, find_lesion_mask


CONFIG_PATH = Path("configs/default.yaml")
TRAIN_FEATURE_PATH = Path("data/processed/features_train.npz")
EDA_DIRECTORY = Path("docs/eda")
TRAIN_KEYS = ("features", "label_ids", "image_ids", "label_names",
              "feature_names", "feature_groups", "feature_regions")
SAMPLE_IMAGES_PER_CLASS = 3
SEGMENTED_IMAGES_PER_CLASS = 2
FIGURE_DPI = 120

SUMMARY_KEYS = (
    "dataset_rows", "image_files", "rows_with_image", "labeled_rows", "usable_images",
    "classes", "class_imbalance_ratio", "multilabel_rows", "labeled_multilabel_rows",
    "constant_columns", "zero_constant_columns", "highly_correlated_pairs",
    "feature_columns", "feature_group_counts", "feature_region_counts", "pca",
    "zero_lesion_block_by_class",
)


def load_label_table(configuration):
    dataset_dir = Path(configuration["data_root"])
    label_table = read_dataset_table(dataset_dir)
    required = ("id", "predominant_stress", "severity", *summaries.DISEASE_COLUMNS)
    summaries.require_columns(label_table, required)
    ids = label_table["id"]
    if not pd.api.types.is_integer_dtype(ids) or not ids.is_unique:
        raise ValueError("CSV image IDs must be unique integers.")
    known_codes = set(configuration["label_by_code"]) | set(configuration["ignored_codes"])
    if not label_table["predominant_stress"].isin(known_codes).all():
        raise ValueError("The CSV contains an unmapped predominant_stress code.")
    image_names = {path.name for path in (dataset_dir / "images").iterdir()
                   if path.suffix.lower() == ".jpg" and path.is_file()}
    expected_names = ids.astype(str) + ".jpg"
    label_table["has_image"] = expected_names.isin(image_names)
    return label_table, len(image_names)


def validate_train_ids(training, sample_count, configuration):
    label_by_code = configuration["label_by_code"]
    expected_labels = [label_by_code[code] for code in sorted(label_by_code)]
    if training["label_names"].tolist() != expected_labels:
        raise ValueError("Archive label_names must match the configured code order.")
    image_ids = training["image_ids"]
    label_ids = training["label_ids"]
    for ids in (image_ids, label_ids):
        if ids.shape != (sample_count,) or ids.dtype.kind not in "iu":
            raise ValueError("Expected one integer image ID and label ID per training row.")
    if np.unique(image_ids).size != sample_count:
        raise ValueError("Train image IDs must be unique.")
    if ((label_ids < 0) | (label_ids >= len(expected_labels))).any():
        raise ValueError("Train label IDs are outside label_names.")


def load_train_features(path, configuration):
    with np.load(path, allow_pickle=False) as archive:
        missing = set(TRAIN_KEYS) - set(archive.files)
        if missing:
            raise ValueError(f"Train archive is missing keys: {sorted(missing)}.")
        training = {key: archive[key] for key in TRAIN_KEYS}
    features = summaries.validate_feature_matrix(training["features"])
    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Expected {EXPECTED_FEATURE_COUNT} feature columns.")
    validate_train_ids(training, len(features), configuration)
    training["labels"] = training["label_names"][training["label_ids"]]
    return training


def build_train_image_paths(label_table, training, configuration):
    train_ids = training["image_ids"]
    rows_by_id = label_table.set_index("id")
    if not pd.Index(train_ids).isin(rows_by_id.index).all():
        raise ValueError("A train image ID is missing from the label CSV.")
    train_rows = rows_by_id.loc[train_ids]
    if not train_rows["has_image"].all():
        raise ValueError("A train image is missing from the image directory.")
    train_labels = train_rows["predominant_stress"].map(configuration["label_by_code"])
    if not np.array_equal(train_labels.to_numpy(), training["labels"]):
        raise ValueError("Train image labels differ between the archive and CSV.")
    image_dir = Path(configuration["data_root"]) / "images"
    return {int(image_id): image_dir / f"{image_id}.jpg" for image_id in train_ids}


def read_train_image_sizes(train_image_paths):
    sizes = []
    for image_path in train_image_paths.values():
        with Image.open(image_path) as image:
            sizes.append(image.size)
    return sizes


def compute_dataset_tables(label_table, training, configuration):
    label_by_code = configuration["label_by_code"]
    overview = summaries.summarize_dataset(label_table, label_by_code)
    split_counts = summaries.summarize_split(overview, training["labels"])
    crosstab, multilabel_counts = summaries.summarize_multilabel(label_table, label_by_code)
    severity = summaries.summarize_severity(label_table, label_by_code)
    return {"dataset_overview": overview, "split_counts": split_counts,
            "multilabel_crosstab": crosstab, "severity_by_class": severity}, multilabel_counts


def compute_feature_tables(training, configuration):
    features = training["features"]
    labels = training["labels"]
    metadata = tuple(training[key] for key in ("feature_names", "feature_groups", "feature_regions"))
    ratios = summaries.extract_lesion_ratios(features, training["feature_names"])
    lesion_stats = summaries.summarize_lesion_ratios(ratios, labels)
    zero_blocks = summaries.summarize_zero_lesion_blocks(features, training["feature_regions"], labels)
    constants = summaries.summarize_constant_columns(features, *metadata)
    correlation, ordered_metadata = summaries.build_correlation_matrix(features, *metadata)
    correlation_stats, top_pairs = summaries.summarize_correlation_pairs(correlation)
    scores, variance, pca_components = summaries.compute_pca(features, configuration["seed"])
    tables = {"lesion_ratio_by_class": lesion_stats, "zero_lesion_block_by_class": zero_blocks,
              "constant_columns": constants, "correlation_summary": correlation_stats,
              "top_correlated_pairs": top_pairs, "pca_variance": variance}
    details = {"ratios": ratios, "correlation": correlation, "ordered_metadata": ordered_metadata,
               "pca_scores": scores, "components_for_90_percent": pca_components}
    return tables, details


def compute_all_tables(label_table, training, configuration):
    train_image_paths = build_train_image_paths(label_table, training, configuration)
    tables, multilabel_counts = compute_dataset_tables(label_table, training, configuration)
    image_sizes = read_train_image_sizes(train_image_paths)
    tables["image_sizes"] = summaries.summarize_image_sizes(image_sizes)
    feature_tables, details = compute_feature_tables(training, configuration)
    tables.update(feature_tables)
    details.update({"train_image_paths": train_image_paths, "multilabel_counts": multilabel_counts})
    return tables, details


def write_tables(tables, output_directory):
    table_directory = output_directory / "tables"
    table_directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        include_index = name == "multilabel_crosstab"
        table.to_csv(table_directory / f"{name}.csv", index=include_index, float_format="%.12g")


def load_sample_images(training, train_image_paths, configuration):
    random_source = np.random.RandomState(configuration["seed"])
    labels = training["labels"]
    samples = {}
    for label in training["label_names"]:
        positions = np.flatnonzero(labels == label)
        if len(positions) < SAMPLE_IMAGES_PER_CLASS:
            raise ValueError(f"Class {label} needs at least {SAMPLE_IMAGES_PER_CLASS} train images.")
        selected = random_source.choice(positions, size=SAMPLE_IMAGES_PER_CLASS, replace=False)
        samples[label] = []
        for position in selected:
            image_id = int(training["image_ids"][position])
            image_path = train_image_paths[image_id]
            raw_image = load_resized_rgb(image_path, configuration["image_max_size"])
            samples[label].append((image_id, raw_image))
    return samples


def segment_sample_images(sample_images):
    segmented_images = {}
    for label, samples in sample_images.items():
        segmented_images[label] = []
        for image_id, raw_image in samples[:SEGMENTED_IMAGES_PER_CLASS]:
            glare_mask = find_glare_mask(raw_image)
            processed_image = preprocess_rgb(raw_image)
            leaf_mask = find_leaf_mask(raw_image)
            lesion_mask = find_lesion_mask(processed_image, leaf_mask, glare_mask)
            segmented_images[label].append((image_id, raw_image, leaf_mask, lesion_mask))
    return segmented_images


def build_all_figures(tables, details, label_table, training, configuration):
    severity = summaries.build_severity_samples(label_table, configuration["label_by_code"])
    samples = load_sample_images(training, details["train_image_paths"], configuration)
    segmented = segment_sample_images(samples)
    ratios = pd.DataFrame({"label": training["labels"], "lesion_ratio": details["ratios"]})
    return {
        "01_class_distribution": plots.plot_class_distribution(tables["split_counts"]),
        "02_multilabel": plots.plot_multilabel(tables["multilabel_crosstab"]),
        "03_severity_by_class": plots.plot_severity_by_class(severity),
        "04_sample_images": plots.plot_sample_images(samples),
        "05_segmentation_overlays": plots.plot_segmentation_overlays(segmented),
        "06_lesion_ratio_by_class": plots.plot_lesion_ratio_by_class(ratios),
        "07_feature_correlation": plots.plot_feature_correlation(details["correlation"], details["ordered_metadata"]),
        "08_pca_scatter": plots.plot_pca_scatter(details["pca_scores"], training["labels"], tables["pca_variance"]),
    }


def write_figures(figures, output_directory):
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        figure.savefig(figure_directory / f"{name}.png", dpi=FIGURE_DPI)
        plt.close(figure)


def build_dataset_summary(tables, image_count):
    overview = tables["dataset_overview"]
    split_counts = tables["split_counts"].set_index("label")
    totals = split_counts["total"]
    return {
        "dataset_rows": int(overview["rows"].sum()),
        "image_files": image_count,
        "rows_with_image": int(overview["images_with_file"].sum()),
        "labeled_rows": int(overview.loc[overview["used"], "rows"].sum()),
        "usable_images": int(totals.sum()),
        "classes": split_counts.to_dict("index"),
        "class_imbalance_ratio": float(totals.max() / totals.min()),
    }


def build_feature_summary(tables, details, training):
    constants = tables["constant_columns"]
    variance = tables["pca_variance"]["explained_variance_ratio"]
    zero_blocks = tables["zero_lesion_block_by_class"].set_index("label")
    return {
        "constant_columns": len(constants),
        "zero_constant_columns": int((constants["value"] == 0).sum()),
        "highly_correlated_pairs": int(tables["correlation_summary"]["high_pairs"].iloc[0]),
        "feature_columns": training["features"].shape[1],
        "feature_group_counts": pd.Series(training["feature_groups"]).value_counts().to_dict(),
        "feature_region_counts": pd.Series(training["feature_regions"]).value_counts().to_dict(),
        "pca": {"explained_variance_ratio": variance.iloc[:2].tolist(),
                "components_for_90_percent": details["components_for_90_percent"]},
        "zero_lesion_block_by_class": zero_blocks["zero_lesion_block"].to_dict(),
    }


def write_summary_json(summary, output_directory):
    missing = set(SUMMARY_KEYS) - set(summary)
    if missing:
        raise ValueError(f"EDA summary is missing keys: {sorted(missing)}.")
    output_directory.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True)
    (output_directory / "eda_summary.json").write_text(contents + "\n", encoding="utf-8")


def main():
    configuration = load_config(CONFIG_PATH)
    label_table, image_count = load_label_table(configuration)
    training = load_train_features(TRAIN_FEATURE_PATH, configuration)
    tables, details = compute_all_tables(label_table, training, configuration)
    write_tables(tables, EDA_DIRECTORY)
    figures = build_all_figures(tables, details, label_table, training, configuration)
    write_figures(figures, EDA_DIRECTORY)
    summary = build_dataset_summary(tables, image_count) | build_feature_summary(tables, details, training)
    summary.update(details["multilabel_counts"])
    write_summary_json(summary, EDA_DIRECTORY)


if __name__ == "__main__":
    plt.switch_backend("Agg")
    with threadpool_limits(limits=1):
        main()
