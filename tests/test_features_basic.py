import numpy as np
import pytest

from coffee_leaf.features.channels import convert_to_gray, convert_to_hsv, convert_to_lab, quantize_channel
from coffee_leaf.features.color import (
    compute_leaf_color_features,
    compute_lesion_color_features,
    list_leaf_color_names,
    list_lesion_color_names,
    normalize_histogram,
)
from coffee_leaf.features.gradient import compute_gradient_features, list_gradient_names
from coffee_leaf.features.masks import build_lesion_roi_mask, shrink_mask
from coffee_leaf.features.shape import compute_shape_features, list_shape_names


def compute_without_mutation(function, *inputs):
    original_arrays = []
    for input_value in inputs:
        if isinstance(input_value, np.ndarray):
            saved_array = input_value.copy()
            original_arrays.append((input_value, saved_array))
    result = function(*inputs)
    for current_array, saved_array in original_arrays:
        np.testing.assert_array_equal(current_array, saved_array)
    return result


def test_convert_color_channels():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    gray_image = compute_without_mutation(convert_to_gray, rgb)
    expected_gray = np.array([[76, 150, 29, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(gray_image, expected_gray)
    assert gray_image.dtype == np.uint8
    hsv_image = compute_without_mutation(convert_to_hsv, rgb)
    hue = hsv_image[:, :, 0]
    np.testing.assert_allclose(hue, [[0, 120, 240, 0]])
    assert hue.min() >= 0
    assert hue.max() <= 360
    lab_image = compute_without_mutation(convert_to_lab, rgb)
    lightness = lab_image[:, :, 0]
    assert lightness.min() >= 0
    assert lightness.max() <= 100


def test_quantize_channel_rounds_and_clips():
    channel = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
    quantized = compute_without_mutation(quantize_channel, channel, -1, 1)
    expected = np.array([0, 0, 128, 255, 255], dtype=np.uint8)
    np.testing.assert_array_equal(quantized, expected)
    assert quantized.dtype == np.uint8


def test_shrink_mask_erodes_all_sides_including_image_border():
    center_mask = np.zeros((9, 9), dtype=bool)
    center_mask[2:7, 2:7] = True
    border_mask = np.zeros((9, 9), dtype=bool)
    border_mask[0:5, 0:5] = True
    expected_center_mask = np.zeros_like(center_mask)
    expected_center_mask[3:6, 3:6] = True
    expected_border_mask = np.zeros_like(border_mask)
    expected_border_mask[1:4, 1:4] = True
    shrunk_center_mask = compute_without_mutation(shrink_mask, center_mask, 3)
    shrunk_border_mask = compute_without_mutation(shrink_mask, border_mask, 3)
    np.testing.assert_array_equal(shrunk_center_mask, expected_center_mask)
    np.testing.assert_array_equal(shrunk_border_mask, expected_border_mask)
    assert shrunk_border_mask.dtype == bool


def test_build_lesion_roi_mask_keeps_only_lesion_on_leaf_without_glare():
    lesion_mask = np.array([[True, True], [True, False]])
    leaf_mask = np.array([[True, True], [False, True]])
    glare_mask = np.array([[False, True], [False, False]])
    roi_mask = compute_without_mutation(build_lesion_roi_mask, lesion_mask, leaf_mask, glare_mask)
    expected_roi_mask = np.array([[True, False], [False, False]])
    np.testing.assert_array_equal(roi_mask, expected_roi_mask)
    assert roi_mask.dtype == bool


def test_feature_vectors_have_expected_lengths_unique_names_and_finite_values():
    random_source = np.random.default_rng(42)
    rgb = random_source.integers(0, 256, size=(10, 12, 3), dtype=np.uint8)
    leaf_mask = np.ones((10, 12), dtype=bool)
    roi_mask = np.zeros_like(leaf_mask)
    roi_mask[2:5, 3:8] = True
    feature_blocks = (
        (compute_lesion_color_features, (rgb, roi_mask), list_lesion_color_names, 102),
        (compute_leaf_color_features, (rgb, leaf_mask), list_leaf_color_names, 100),
        (compute_shape_features, (roi_mask, leaf_mask), list_shape_names, 7),
        (compute_gradient_features, (rgb, leaf_mask), list_gradient_names, 24),
    )
    all_feature_names = []
    for compute_features, inputs, list_names, expected_length in feature_blocks:
        vector = compute_without_mutation(compute_features, *inputs)
        repeated_vector = compute_without_mutation(compute_features, *inputs)
        feature_names = list_names()
        assert vector.shape == (expected_length,)
        assert vector.dtype == np.float64
        assert np.isfinite(vector).all()
        assert len(feature_names) == expected_length
        np.testing.assert_array_equal(vector, repeated_vector)
        all_feature_names.extend(feature_names)
    assert len(all_feature_names) == len(set(all_feature_names))


def test_feature_names_follow_channel_and_bin_order():
    lesion_names = list_lesion_color_names()
    assert lesion_names[0:96:16] == [
        "lesion_hsv_h_00", "lesion_hsv_s_00", "lesion_hsv_v_00",
        "lesion_lab_l_00", "lesion_lab_a_00", "lesion_lab_b_00",
    ]
    assert lesion_names[96:] == [
        "lesion_lab_mean_l", "lesion_lab_mean_a", "lesion_lab_mean_b",
        "lesion_lab_std_l", "lesion_lab_std_a", "lesion_lab_std_b",
    ]
    leaf_names = list_leaf_color_names()
    assert leaf_names[9:11] == ["leaf_ab_hist_a00_b09", "leaf_ab_hist_a01_b00"]
    assert leaf_names[-1] == "leaf_ab_hist_a09_b09"
    gradient_names = list_gradient_names()
    assert gradient_names[::8] == ["leaf_grad_l_00", "leaf_grad_a_00", "leaf_grad_b_00"]
    assert gradient_names[-1] == "leaf_grad_b_07"


def test_color_histograms_for_one_color_have_a_single_filled_bin():
    rgb = np.full((7, 7, 3), 128, dtype=np.uint8)
    leaf_mask = np.ones((7, 7), dtype=bool)
    lesion_features = compute_lesion_color_features(rgb, leaf_mask)
    lesion_histograms = lesion_features[:96].reshape(6, 16)
    for histogram in lesion_histograms:
        assert np.count_nonzero(histogram) == 1
        assert histogram.max() == 1
    leaf_histogram = compute_leaf_color_features(rgb, leaf_mask)
    assert np.count_nonzero(leaf_histogram) == 1
    assert leaf_histogram.max() == 1


def test_gradient_histograms_for_flat_image_fill_only_the_first_bin():
    rgb = np.full((7, 7, 3), 128, dtype=np.uint8)
    leaf_mask = np.ones((7, 7), dtype=bool)
    histogram_vector = compute_gradient_features(rgb, leaf_mask)
    expected_histograms = np.zeros((3, 8))
    expected_histograms[:, 0] = 1
    expected_vector = expected_histograms.ravel()
    np.testing.assert_array_equal(histogram_vector, expected_vector)


def test_empty_histograms_normalize_to_zero():
    histogram_counts = np.zeros(8, dtype=int)
    normalized_histogram = compute_without_mutation(normalize_histogram, histogram_counts)
    np.testing.assert_array_equal(normalized_histogram, np.zeros(8))
    rgb = np.full((5, 5, 3), 128, dtype=np.uint8)
    leaf_mask = np.zeros((5, 5), dtype=bool)
    leaf_mask[2, 2] = True
    gradient_features = compute_gradient_features(rgb, leaf_mask)
    np.testing.assert_array_equal(gradient_features, np.zeros(24))


def test_shape_features_for_two_square_regions():
    leaf_mask = np.ones((10, 10), dtype=bool)
    roi_mask = np.zeros_like(leaf_mask)
    roi_mask[1:3, 1:3] = True
    roi_mask[5:8, 6:9] = True
    shape_features = compute_shape_features(roi_mask, leaf_mask)
    expected = np.array([2, 0.065, 0.09, 0, 1, 35 / 130, 0.13])
    np.testing.assert_allclose(shape_features, expected)


def test_shape_features_count_diagonal_pixels_separately():
    leaf_mask = np.ones((5, 5), dtype=bool)
    roi_mask = np.zeros_like(leaf_mask)
    roi_mask[0, 0] = True
    roi_mask[1, 1] = True
    shape_features = compute_shape_features(roi_mask, leaf_mask)
    expected = np.array([2, 0.04, 0.04, 0, 1, 0.3, 0.08])
    np.testing.assert_allclose(shape_features, expected)


def test_shape_features_for_empty_roi_are_zero():
    leaf_mask = np.ones((5, 5), dtype=bool)
    roi_mask = np.zeros_like(leaf_mask)
    shape_features = compute_shape_features(roi_mask, leaf_mask)
    np.testing.assert_array_equal(shape_features, np.zeros(7))


def test_features_reject_invalid_shapes_and_empty_required_masks():
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    leaf_mask = np.ones((5, 5), dtype=bool)
    empty_mask = np.zeros_like(leaf_mask)
    wrong_shape_mask = np.ones((4, 5), dtype=bool)
    with pytest.raises(ValueError, match="roi_mask must have shape"):
        compute_lesion_color_features(rgb, wrong_shape_mask)
    with pytest.raises(ValueError, match="roi_mask must contain"):
        compute_lesion_color_features(rgb, empty_mask)
    with pytest.raises(ValueError, match="leaf_mask must contain"):
        compute_leaf_color_features(rgb, empty_mask)
    with pytest.raises(ValueError, match="leaf_mask must contain"):
        compute_gradient_features(rgb, empty_mask)
    with pytest.raises(ValueError, match="leaf_mask must contain"):
        compute_shape_features(empty_mask, empty_mask)
    with pytest.raises(ValueError, match="glare_mask must have shape"):
        build_lesion_roi_mask(leaf_mask, leaf_mask, wrong_shape_mask)
    with pytest.raises(ValueError, match="dtype uint8"):
        compute_gradient_features(rgb.astype(float), leaf_mask)
