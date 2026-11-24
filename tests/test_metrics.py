import pandas as pd
import pytest

from coffee_leaf.evaluation.metrics import (
    build_confusion_matrix_table,
    build_per_class_table,
    compute_accuracy_and_macro_f1,
)


@pytest.fixture
def classification_labels():
    true_label_ids = [0, 0, 1, 1, 2, 2]
    predicted_label_ids = [0, 0, 0, 1, 1, 2]
    class_ids = [0, 1, 2]
    class_names = ["zeta", "alpha", "delta"]
    return true_label_ids, predicted_label_ids, class_ids, class_names


def test_accuracy_and_macro_f1_match_hand_calculation(classification_labels):
    true_label_ids, predicted_label_ids, class_ids, class_names = classification_labels
    accuracy, macro_f1 = compute_accuracy_and_macro_f1(true_label_ids, predicted_label_ids, class_ids)
    assert accuracy == pytest.approx(4 / 6)
    assert macro_f1 == pytest.approx(59 / 90)


def test_confusion_matrix_preserves_class_order(classification_labels):
    matrix = build_confusion_matrix_table(*classification_labels)
    class_names = classification_labels[-1]
    expected_matrix = pd.DataFrame(
        [[2, 0, 0], [1, 1, 0], [0, 1, 1]],
        index=pd.Index(class_names, name="actual_class"),
        columns=class_names,
    )
    pd.testing.assert_frame_equal(matrix, expected_matrix)


def test_per_class_table_matches_hand_calculation(classification_labels):
    per_class_table = build_per_class_table(*classification_labels)
    expected_table = pd.DataFrame({
        "class_name": classification_labels[-1],
        "precision": [2 / 3, 1 / 2, 1],
        "recall": [1, 1 / 2, 1 / 2],
        "f1": [4 / 5, 1 / 2, 2 / 3],
        "support": [2, 2, 2],
    })
    pd.testing.assert_frame_equal(per_class_table, expected_table)
