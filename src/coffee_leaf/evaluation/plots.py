import matplotlib.pyplot as plt
import numpy as np


def draw_validation_axis(axis, positions, labels, scores, title):
    train_scores, cv_scores, cv_std = scores
    bar_width = 0.35
    axis.bar(positions - bar_width / 2, train_scores, bar_width, label="Train")
    axis.bar(positions + bar_width / 2, cv_scores, bar_width, yerr=cv_std, label="Validation (CV)")
    axis.set_title(title)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1.05)
    axis.legend()


def plot_validation_scores(cv_results):
    candidate_labels = cv_results["model_name"] + "\n" + cv_results["class_weight"]
    candidate_positions = np.arange(len(cv_results))
    accuracy_scores = cv_results["train_accuracy"], cv_results["cv_accuracy"], cv_results["cv_accuracy_std"]
    f1_scores = cv_results["train_macro_f1"], cv_results["best_cv_macro_f1"], cv_results["cv_std"]
    figure, (accuracy_axis, f1_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    draw_validation_axis(accuracy_axis, candidate_positions, candidate_labels, accuracy_scores, "Accuracy")
    draw_validation_axis(f1_axis, candidate_positions, candidate_labels, f1_scores, "Macro-F1")
    figure.tight_layout()
    return figure


def draw_matrix(axis, matrix, class_names, title, number_format):
    class_count = len(class_names)
    axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xticks(range(class_count))
    axis.set_yticks(range(class_count))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)
    axis.xaxis.set_label_text("Predicted")
    axis.set_ylabel("True")
    threshold = matrix.max() / 2
    for row_index in range(class_count):
        for column_index in range(class_count):
            value = matrix[row_index, column_index]
            text_color = "white" if value > threshold else "black"
            axis.text(column_index, row_index, format(value, number_format), ha="center", va="center", color=text_color)


def plot_confusion_matrices(confusion_matrix_table):
    class_names = confusion_matrix_table.index.tolist()
    confusion_counts = confusion_matrix_table.to_numpy()
    row_totals = confusion_counts.sum(axis=1, keepdims=True)
    confusion_rates = confusion_counts / np.maximum(row_totals, 1)
    figure, (counts_axis, rates_axis) = plt.subplots(1, 2, figsize=(13, 5.5))
    draw_matrix(counts_axis, confusion_counts, class_names, "Counts", "d")
    draw_matrix(rates_axis, confusion_rates, class_names, "Row-normalized", ".2f")
    figure.tight_layout()
    return figure


def draw_per_class_scores(axis, per_class_table):
    metric_names = ["precision", "recall", "f1"]
    class_positions = np.arange(len(per_class_table))
    bar_width = 0.25
    for metric_index, metric_name in enumerate(metric_names):
        offset = (metric_index - 1) * bar_width
        axis.bar(class_positions + offset, per_class_table[metric_name], bar_width, label=metric_name)
    axis.set_xticks(class_positions)
    axis.set_xticklabels(per_class_table["class_name"])
    axis.set_ylim(0, 1.05)
    axis.set_title("Per-class scores (cross-validation on train)")
    axis.legend()


def plot_per_class_scores(per_class_table):
    figure, axis = plt.subplots(figsize=(10, 4.5))
    draw_per_class_scores(axis, per_class_table)
    figure.tight_layout()
    return figure


def draw_importance_axis(axis, importance_table, label_column, title):
    axis.barh(importance_table[label_column], importance_table["importance"])
    axis.invert_yaxis()
    axis.set_title(title)
    axis.xaxis.set_label_text("Importance")


def plot_tree_importance(feature_importance_table, group_importance, region_importance):
    top_features = feature_importance_table.head(15)
    figure, (feature_axis, group_axis, region_axis) = plt.subplots(1, 3, figsize=(16, 6))
    draw_importance_axis(feature_axis, top_features, "feature_name", "Top 15 features")
    draw_importance_axis(group_axis, group_importance, "group", "By group")
    draw_importance_axis(region_axis, region_importance, "region", "By region")
    figure.tight_layout()
    return figure
