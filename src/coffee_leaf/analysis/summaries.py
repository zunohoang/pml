import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DISEASE_COLUMNS = ("miner", "rust", "phoma", "cercospora")
LESION_RATIO_COLUMN = "lesion_shape_area_ratio"
HIGH_CORRELATION_THRESHOLD = 0.9
TOP_CORRELATED_PAIR_COUNT = 20
PCA_VARIANCE_TARGET = 0.9


def require_columns(table, columns):
    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("Expected a nonempty DataFrame.")
    missing_columns = set(columns) - set(table.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}.")


def validate_labels(labels, sample_count):
    labels = np.asarray(labels)
    if labels.shape != (sample_count,) or not sample_count:
        raise ValueError("Expected one class label per sample.")
    if labels.dtype.kind != "U" or (labels == "").any():
        raise ValueError("Class labels must be nonempty strings.")
    return labels


def summarize_dataset(label_table, label_by_code):
    require_columns(label_table, ("predominant_stress", "has_image"))
    if not pd.api.types.is_integer_dtype(label_table["predominant_stress"]):
        raise ValueError("predominant_stress must contain integer codes.")
    if label_table["has_image"].dtype != bool:
        raise ValueError("has_image must contain boolean flags.")
    grouped_rows = label_table.groupby("predominant_stress", as_index=False)
    overview = grouped_rows.agg(rows=("has_image", "size"), images_with_file=("has_image", "sum"))
    codes = overview["predominant_stress"]
    overview["label"] = codes.map(label_by_code).fillna("ignored")
    overview["used"] = codes.isin(label_by_code)
    overview["usable_images"] = overview["images_with_file"].where(overview["used"], 0)
    return overview


def summarize_split(dataset_overview, train_labels):
    require_columns(dataset_overview, ("label", "used", "usable_images"))
    train_labels = np.asarray(train_labels)
    train_labels = validate_labels(train_labels, train_labels.size)
    usable_rows = dataset_overview.loc[dataset_overview["used"], ["label", "usable_images"]]
    split_counts = usable_rows.rename(columns={"usable_images": "total"}).reset_index(drop=True)
    train_counts = pd.Series(train_labels).value_counts()
    if split_counts.empty or not train_counts.index.isin(split_counts["label"]).all():
        raise ValueError("Train labels must be present in the dataset overview.")
    if not split_counts["label"].is_unique or (split_counts["total"] <= 0).any():
        raise ValueError("Expected unique labels with positive class totals.")
    split_counts["train"] = split_counts["label"].map(train_counts).fillna(0).astype(int)
    split_counts["test"] = split_counts["total"] - split_counts["train"]
    if (split_counts["test"] < 0).any():
        raise ValueError("Train count cannot exceed the available images in any class.")
    return split_counts


def summarize_multilabel(label_table, label_by_code):
    require_columns(label_table, ("predominant_stress", *DISEASE_COLUMNS))
    disease_flags = label_table[list(DISEASE_COLUMNS)]
    if not disease_flags.isin((0, 1)).to_numpy().all():
        raise ValueError("Disease flags must contain only 0 and 1.")
    codes = label_table["predominant_stress"]
    if not pd.api.types.is_integer_dtype(codes):
        raise ValueError("predominant_stress must contain integer codes.")
    flag_strings = disease_flags.astype(int).astype(str)
    patterns = flag_strings.agg("".join, axis=1).rename("disease_pattern")
    crosstab = pd.crosstab(codes, patterns)
    multilabel_mask = disease_flags.sum(axis=1) >= 2
    labeled_mask = codes.isin(label_by_code)
    counts = {
        "multilabel_rows": int(multilabel_mask.sum()),
        "labeled_multilabel_rows": int((multilabel_mask & labeled_mask).sum()),
    }
    return crosstab, counts


def summarize_class_values(values, labels, value_name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError(f"{value_name} must be a finite numeric vector.")
    labels = validate_labels(labels, values.size)
    samples = pd.DataFrame({"label": labels, value_name: values})
    grouped = samples.groupby("label", sort=False)[value_name]
    summary = grouped.agg(n="size", minimum="min", mean="mean", median="median", maximum="max")
    summary["q1"] = grouped.quantile(0.25)
    summary["q3"] = grouped.quantile(0.75)
    summary = summary.reset_index()
    return summary[["label", "n", "minimum", "q1", "median", "mean", "q3", "maximum"]]


def build_severity_samples(label_table, label_by_code):
    require_columns(label_table, ("predominant_stress", "severity"))
    mapped_labels = label_table["predominant_stress"].map(label_by_code)
    labeled_mask = mapped_labels.notna()
    severity = label_table.loc[labeled_mask, "severity"]
    if severity.empty or not pd.api.types.is_numeric_dtype(severity) or not np.isfinite(severity).all():
        raise ValueError("Labeled severity values must be numeric and finite.")
    return pd.DataFrame({"label": mapped_labels[labeled_mask], "severity": severity})


def summarize_severity(label_table, label_by_code):
    samples = build_severity_samples(label_table, label_by_code)
    return summarize_class_values(samples["severity"], samples["label"].to_numpy(dtype=str), "severity")


def summarize_image_sizes(dimensions):
    dimensions = np.asarray(dimensions)
    if dimensions.ndim != 2 or dimensions.shape[1] != 2 or not len(dimensions):
        raise ValueError("Expected a nonempty array of image widths and heights.")
    if dimensions.dtype.kind not in "iu" or (dimensions <= 0).any():
        raise ValueError("Image widths and heights must be positive integers.")
    sizes = pd.DataFrame(dimensions, columns=["width", "height"])
    return sizes.value_counts(sort=False).rename("image_count").reset_index()


def validate_feature_matrix(features):
    features = np.asarray(features)
    if features.ndim != 2 or min(features.shape) == 0:
        raise ValueError("Expected a nonempty two-dimensional feature matrix.")
    if features.dtype.kind not in "iuf" or not np.isfinite(features).all():
        raise ValueError("Feature matrix must contain only finite real values.")
    return features


def build_feature_metadata(features, feature_names, feature_groups, feature_regions):
    features = validate_feature_matrix(features)
    column_count = features.shape[1]
    names = validate_labels(feature_names, column_count)
    groups = validate_labels(feature_groups, column_count)
    regions = validate_labels(feature_regions, column_count)
    if np.unique(names).size != column_count:
        raise ValueError("Feature names must be unique.")
    return pd.DataFrame({"feature_name": names, "feature_group": groups, "feature_region": regions})


def extract_lesion_ratios(features, feature_names):
    features = validate_feature_matrix(features)
    names = validate_labels(feature_names, features.shape[1])
    ratio_indices = np.flatnonzero(names == LESION_RATIO_COLUMN)
    if ratio_indices.size != 1:
        raise ValueError(f"Expected exactly one {LESION_RATIO_COLUMN} column.")
    ratios = features[:, ratio_indices[0]]
    if ((ratios < 0) | (ratios > 1)).any():
        raise ValueError("Lesion area ratios must lie between 0 and 1.")
    return ratios


def summarize_lesion_ratios(ratios, labels):
    ratios = np.asarray(ratios)
    if ratios.dtype.kind not in "iuf" or ((ratios < 0) | (ratios > 1)).any():
        raise ValueError("Lesion area ratios must lie between 0 and 1.")
    return summarize_class_values(ratios, labels, "lesion_ratio")


def summarize_zero_lesion_blocks(features, feature_regions, labels):
    features = validate_feature_matrix(features)
    regions = validate_labels(feature_regions, features.shape[1])
    labels = validate_labels(labels, len(features))
    lesion_mask = regions == "lesion"
    if not lesion_mask.any():
        raise ValueError("Feature metadata must include lesion columns.")
    zero_block_mask = (features[:, lesion_mask] == 0).all(axis=1)
    rows = pd.DataFrame({"label": labels, "zero_lesion_block": zero_block_mask})
    grouped = rows.groupby("label", as_index=False, sort=False)
    return grouped.agg(n=("zero_lesion_block", "size"), zero_lesion_block=("zero_lesion_block", "sum"))


def summarize_constant_columns(features, feature_names, feature_groups, feature_regions):
    metadata = build_feature_metadata(features, feature_names, feature_groups, feature_regions)
    features = np.asarray(features)
    constant_mask = (features == features[0]).all(axis=0)
    constant_columns = metadata.loc[constant_mask].copy()
    constant_columns["value"] = features[0, constant_mask]
    return constant_columns.reset_index(drop=True)


def build_correlation_matrix(features, feature_names, feature_groups, feature_regions):
    metadata = build_feature_metadata(features, feature_names, feature_groups, feature_regions)
    features = np.asarray(features)
    varying_mask = (features != features[0]).any(axis=0)
    varying_columns = metadata.loc[varying_mask]
    if len(varying_columns) < 2:
        raise ValueError("Correlation requires at least two nonconstant columns.")
    ordered = varying_columns.sort_values(["feature_group", "feature_region"], kind="stable")
    values = features[:, ordered.index]
    correlation = np.corrcoef(values, rowvar=False)
    if not np.isfinite(correlation).all():
        raise ValueError("Feature correlation must be finite.")
    correlation = np.clip(correlation, -1, 1)
    names = ordered["feature_name"].tolist()
    return pd.DataFrame(correlation, index=names, columns=names), ordered.reset_index(drop=True)


def validate_correlation_matrix(correlation):
    require_columns(correlation, ())
    if not correlation.index.equals(correlation.columns) or not correlation.columns.is_unique:
        raise ValueError("Correlation row and column names must match in order and be unique.")
    values = correlation.to_numpy()
    if values.shape[0] != values.shape[1] or len(values) < 2:
        raise ValueError("Expected a square correlation matrix with at least two columns.")
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("Expected a finite square correlation matrix with at least two columns.")
    return values


def summarize_correlation_pairs(correlation):
    values = validate_correlation_matrix(correlation)
    first, second = np.triu_indices(len(values), k=1)
    coefficients = values[first, second]
    magnitudes = np.abs(coefficients)
    strong_mask = magnitudes >= HIGH_CORRELATION_THRESHOLD
    linked = np.union1d(first[strong_mask], second[strong_mask])
    summary = pd.DataFrame([{
        "columns": len(values), "pairs": len(coefficients), "high_pairs": int(strong_mask.sum()),
        "linked_columns": len(linked), "linked_share": len(linked) / len(values),
        "threshold": HIGH_CORRELATION_THRESHOLD,
    }])
    order = np.argsort(-magnitudes, kind="stable")[:TOP_CORRELATED_PAIR_COUNT]
    names = correlation.columns.to_numpy()
    pairs = pd.DataFrame({"first_feature": names[first[order]], "second_feature": names[second[order]],
                          "r": coefficients[order], "absolute_r": magnitudes[order]})
    return summary, pairs


def compute_pca(features, seed):
    features = validate_feature_matrix(features)
    if min(features.shape) < 2 or not (features != features[0]).any():
        raise ValueError("PCA needs at least two rows and columns and some nonconstant values.")
    scaled = StandardScaler().fit_transform(features)
    model = PCA(svd_solver="full", random_state=seed)
    scores = model.fit_transform(scaled)
    explained = model.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    variance = pd.DataFrame({"component": np.arange(1, len(explained) + 1),
                             "explained_variance_ratio": explained, "cumulative_variance": cumulative})
    components_for_target = int(np.searchsorted(cumulative, PCA_VARIANCE_TARGET) + 1)
    return scores[:, :2], variance, components_for_target
