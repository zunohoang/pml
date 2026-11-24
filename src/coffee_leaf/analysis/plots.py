import matplotlib.pyplot as plt
import numpy as np


SPLIT_COLORS = ("steelblue", "seagreen", "peru")
LEAF_OVERLAY_COLOR = (0.12, 0.76, 0.38, 0.28)
LESION_OVERLAY_COLOR = (1.0, 0.38, 0.10, 0.60)


def plot_class_distribution(split_counts):
    figure, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
    positions = np.arange(len(split_counts))
    width = 0.25
    for index, column in enumerate(("total", "train", "test")):
        offset = (index - 1) * width
        axis.bar(positions + offset, split_counts[column], width, color=SPLIT_COLORS[index], label=column.title())
    totals = split_counts["total"]
    imbalance = totals.max() / totals.min()
    axis.set_xticks(positions, split_counts["label"], rotation=20, ha="right")
    axis.yaxis.set_label_text("Images")
    axis.set_title(f"Class distribution · largest / smallest = {imbalance:.2f}")
    axis.legend()
    return figure


def plot_multilabel(crosstab):
    figure, axis = plt.subplots(figsize=(10, 4.5), layout="constrained")
    values = crosstab.to_numpy()
    heatmap = axis.imshow(values, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(crosstab.columns)), crosstab.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(crosstab.index)), crosstab.index)
    axis.xaxis.set_label_text("Disease flags (miner, rust, phoma, cercospora)")
    axis.yaxis.set_label_text("Predominant stress code")
    axis.set_title("Disease flags by predominant stress")
    midpoint = values.max() / 2
    for (row, column), count in np.ndenumerate(values):
        color = "white" if count > midpoint else "black"
        axis.text(column, row, str(count), ha="center", va="center", color=color, fontsize=8)
    figure.colorbar(heatmap, ax=axis, label="CSV rows")
    return figure


def plot_class_boxplots(samples, value_name, title, unit):
    figure, axis = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    groups = list(samples.groupby("label", sort=False)[value_name])
    labels = [label for label, group in groups]
    values = [group.to_numpy() for label, group in groups]
    axis.boxplot(values, patch_artist=True, boxprops={"facecolor": "lightblue"})
    positions = np.arange(1, len(groups) + 1)
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.yaxis.set_label_text(unit)
    axis.set_title(title)
    return figure


def plot_severity_by_class(samples):
    return plot_class_boxplots(samples, "severity", "Severity by class · labeled CSV rows", "Recorded severity")


def draw_sample_image(axis, label, image_id, image):
    axis.imshow(image)
    axis.set_title(f"{label} · {image_id}", fontsize=9)
    axis.set_axis_off()


def plot_sample_images(sample_images):
    column_count = len(next(iter(sample_images.values())))
    figure, axes = plt.subplots(len(sample_images), column_count, squeeze=False,
                                figsize=(3 * column_count, 1.7 * len(sample_images)), layout="constrained")
    figure.suptitle("Seeded training images")
    for axes_row, (label, images) in zip(axes, sample_images.items()):
        for axis, (image_id, image) in zip(axes_row, images):
            draw_sample_image(axis, label, image_id, image)
    return figure


def draw_segmentation_pair(raw_axis, overlay_axis, label, sample):
    image_id, raw_image, leaf_mask, lesion_mask = sample
    draw_sample_image(raw_axis, label, image_id, raw_image)
    draw_sample_image(overlay_axis, "Overlay", image_id, raw_image)
    overlay = np.zeros((*leaf_mask.shape, 4))
    overlay[leaf_mask] = LEAF_OVERLAY_COLOR
    overlay[lesion_mask] = LESION_OVERLAY_COLOR
    overlay_axis.imshow(overlay)


def plot_segmentation_overlays(segmented_images):
    column_count = 2 * len(next(iter(segmented_images.values())))
    figure, axes = plt.subplots(len(segmented_images), column_count, squeeze=False,
                                figsize=(2.6 * column_count, 1.7 * len(segmented_images)), layout="constrained")
    figure.suptitle("Segmentation · leaf (green) and suspected lesion (orange)")
    for axes_row, (label, images) in zip(axes, segmented_images.items()):
        for position, image in enumerate(images):
            draw_segmentation_pair(axes_row[2 * position], axes_row[2 * position + 1], label, image)
    return figure


def plot_lesion_ratio_by_class(samples):
    return plot_class_boxplots(samples, "lesion_ratio", "Suspected lesion area · training images", "Fraction of leaf")


def draw_correlation_blocks(axis, metadata):
    groups = metadata["feature_group"] + " / " + metadata["feature_region"]
    boundaries = np.flatnonzero(groups.to_numpy()[1:] != groups.to_numpy()[:-1]) + 1
    edges = np.concatenate(([0], boundaries, [len(groups)]))
    centers = (edges[1:] + edges[:-1] - 1) / 2
    labels = groups.iloc[edges[:-1]].tolist()
    axis.set_xticks(centers, labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(centers, labels, fontsize=8)
    for boundary in boundaries:
        axis.axvline(boundary - 0.5, color="black", linewidth=0.6)
        axis.axhline(boundary - 0.5, color="black", linewidth=0.6)


def plot_feature_correlation(correlation, ordered_metadata):
    figure, axis = plt.subplots(figsize=(8, 7), layout="constrained")
    heatmap = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    draw_correlation_blocks(axis, ordered_metadata)
    axis.set_title("Pearson correlation · nonconstant training features")
    figure.colorbar(heatmap, ax=axis, label="Pearson r")
    return figure


def plot_pca_scatter(scores, train_labels, variance):
    figure, axis = plt.subplots(figsize=(8, 5.5), layout="constrained")
    for label in np.unique(train_labels):
        class_mask = train_labels == label
        class_scores = scores[class_mask]
        axis.scatter(class_scores[:, 0], class_scores[:, 1], s=14, alpha=0.6, label=label)
    explained = variance["explained_variance_ratio"].to_numpy()
    axis.xaxis.set_label_text(f"PC1 ({explained[0]:.1%})")
    axis.yaxis.set_label_text(f"PC2 ({explained[1]:.1%})")
    axis.set_title("PCA of standardized training features")
    axis.legend()
    return figure
