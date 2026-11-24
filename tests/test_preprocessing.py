import numpy as np

from coffee_leaf.config import load_config
from coffee_leaf.data.loader import build_image_table
from coffee_leaf.preprocessing.image import balance_white, find_glare_mask, load_resized_rgb, preprocess_rgb
from coffee_leaf.preprocessing.segmentation import find_leaf_mask, find_lesion_mask


def load_sample_rgb():
    config = load_config("configs/default.yaml")
    image_table = build_image_table(config)
    rust_rows = image_table["label"] == "rust"
    rust_images = image_table.loc[rust_rows, "image_path"]
    image_path = rust_images.iloc[0]
    max_side = config["image_max_size"]
    return load_resized_rgb(image_path, max_side)


def compute_preprocessing_outputs(raw_rgb):
    processed_rgb = preprocess_rgb(raw_rgb)
    glare_mask = find_glare_mask(raw_rgb)
    leaf_mask = find_leaf_mask(raw_rgb)
    lesion_mask = find_lesion_mask(processed_rgb, leaf_mask, glare_mask)
    return processed_rgb, glare_mask, leaf_mask, lesion_mask


def test_load_resized_rgb_returns_uint8_rgb():
    raw_rgb = load_sample_rgb()
    assert raw_rgb.shape == (128, 256, 3)
    assert raw_rgb.dtype == np.uint8


def test_preprocess_rgb_preserves_shape_and_dtype():
    raw_rgb = load_sample_rgb()
    processed_rgb = preprocess_rgb(raw_rgb)
    assert processed_rgb.shape == raw_rgb.shape
    assert processed_rgb.dtype == raw_rgb.dtype


def test_balance_white_reduces_channel_mean_spread():
    raw_rgb = load_sample_rgb()
    tinted_values = raw_rgb.astype(float)
    tinted_values[..., 0] *= 1.3
    clipped_values = np.clip(tinted_values, 0, 255)
    tinted_rgb = clipped_values.astype(np.uint8)
    balanced_rgb = balance_white(tinted_rgb)
    tinted_means = tinted_rgb.mean(axis=(0, 1))
    balanced_means = balanced_rgb.mean(axis=(0, 1))
    tinted_spread = np.ptp(tinted_means)
    balanced_spread = np.ptp(balanced_means)
    assert balanced_spread < tinted_spread


def test_find_leaf_mask_on_two_images_per_class():
    config = load_config("configs/default.yaml")
    image_table = build_image_table(config)
    images_by_class = image_table.groupby("label")
    sample_table = images_by_class.head(2)
    assert len(sample_table) == 10
    max_side = config["image_max_size"]
    for image_row in sample_table.itertuples(index=False):
        raw_rgb = load_resized_rgb(image_row.image_path, max_side)
        leaf_mask = find_leaf_mask(raw_rgb)
        assert leaf_mask.dtype == bool
        assert leaf_mask.shape == raw_rgb.shape[:2]
        leaf_fraction = leaf_mask.mean()
        assert 0.05 <= leaf_fraction <= 0.90, image_row.id


def test_find_lesion_mask_respects_leaf_and_glare_masks():
    raw_rgb = load_sample_rgb()
    _, glare_mask, leaf_mask, lesion_mask = compute_preprocessing_outputs(raw_rgb)
    outside_leaf = lesion_mask & ~leaf_mask
    overlapping_glare = lesion_mask & glare_mask
    assert lesion_mask.dtype == bool
    assert lesion_mask.shape == leaf_mask.shape
    assert not outside_leaf.any()
    assert not overlapping_glare.any()


def test_preprocessing_repeats_with_the_same_input():
    raw_rgb = load_sample_rgb()
    first_outputs = compute_preprocessing_outputs(raw_rgb)
    repeated_outputs = compute_preprocessing_outputs(raw_rgb)
    for first_output, repeated_output in zip(first_outputs, repeated_outputs):
        np.testing.assert_array_equal(first_output, repeated_output)
