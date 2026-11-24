from pandas import RangeIndex
from pandas.testing import assert_frame_equal

from coffee_leaf.config import load_config
from coffee_leaf.data.loader import build_image_table
from coffee_leaf.data.split import split_train_test


def test_build_image_table():
    config = load_config("configs/default.yaml")
    image_table = build_image_table(config)
    columns = list(image_table.columns)
    assert columns == ["id", "label", "image_path"]
    assert len(image_table) == 1342
    expected_counts = {
        "healthy": 142,
        "leaf_miner": 253,
        "rust": 465,
        "brown_spot": 346,
        "cercospora": 136,
    }
    label_counts = image_table["label"].value_counts()
    assert label_counts.to_dict() == expected_counts
    assert image_table["id"].is_unique
    for image_path in image_table["image_path"]:
        assert image_path.is_file()


def test_split_train_test():
    config = load_config("configs/default.yaml")
    image_table = build_image_table(config)
    train_table, test_table = split_train_test(image_table, config)
    train_count = len(train_table)
    test_count = len(test_table)
    total_count = train_count + test_count
    assert total_count == 1342
    train_ids = set(train_table["id"])
    test_ids = set(test_table["id"])
    assert train_ids.isdisjoint(test_ids)
    expected_labels = set(config["classes"])
    train_labels = set(train_table["label"])
    test_labels = set(test_table["label"])
    assert train_labels == expected_labels
    assert test_labels == expected_labels
    test_ratio = test_count / total_count
    assert 0.24 <= test_ratio <= 0.26


def test_split_train_test_repeats_with_the_same_seed():
    config = load_config("configs/default.yaml")
    image_table = build_image_table(config)
    train_table, test_table = split_train_test(image_table, config)
    repeated_train_table, repeated_test_table = split_train_test(image_table, config)
    assert_frame_equal(train_table, repeated_train_table)
    assert_frame_equal(test_table, repeated_test_table)
    train_count = len(train_table)
    test_count = len(test_table)
    expected_train_index = RangeIndex(train_count)
    expected_test_index = RangeIndex(test_count)
    assert train_table.index.equals(expected_train_index)
    assert test_table.index.equals(expected_test_index)
