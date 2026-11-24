import numpy as np

from coffee_leaf.features.texture import (
    compute_glcm_features,
    compute_glcm_properties_for_offset,
    compute_lbp_histogram,
    compute_leaf_glcm_features,
    compute_leaf_lbp_features,
    compute_lesion_texture_features,
    list_leaf_glcm_names,
    list_leaf_lbp_names,
    list_lesion_texture_names,
    list_valid_pairs,
    reduce_to_levels,
)


def test_reduce_to_levels_uses_sixteen_equal_bins():
    channel = np.array([[0, 15, 16, 31, 32, 239, 240, 255]], dtype=np.uint8)
    levels = reduce_to_levels(channel)
    expected = np.array([[0, 0, 1, 1, 2, 14, 15, 15]], dtype=np.uint8)
    np.testing.assert_array_equal(levels, expected)
    assert levels.dtype == np.uint8


def test_list_valid_pairs_for_horizontal_shift_and_mask_hole():
    level_image = np.arange(9, dtype=np.uint8).reshape(3, 3)
    mask = np.ones((3, 3), dtype=bool)
    full_pairs = list_valid_pairs(level_image, mask, 0, 1)
    expected_pairs = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8]]
    np.testing.assert_array_equal(full_pairs, expected_pairs)
    mask[1, 1] = False
    masked_pairs = list_valid_pairs(level_image, mask, 0, 1)
    expected_masked_pairs = [[0, 1], [1, 2], [6, 7], [7, 8]]
    np.testing.assert_array_equal(masked_pairs, expected_masked_pairs)


def test_list_valid_pairs_for_negative_row_shift():
    level_image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    mask = np.ones((4, 4), dtype=bool)
    pairs = list_valid_pairs(level_image, mask, -1, 1)
    expected = [[4, 1], [5, 2], [6, 3], [8, 5], [9, 6], [10, 7], [12, 9], [13, 10], [14, 11]]
    np.testing.assert_array_equal(pairs, expected)


def test_glcm_of_flat_image_has_zero_contrast_and_unit_correlation():
    level_image = np.full((5, 5), 3, dtype=np.uint8)
    mask = np.ones((5, 5), dtype=bool)
    properties = compute_glcm_features(level_image, mask, 1)
    expected_properties = [0, 0, 1, 1, 1, 1]
    np.testing.assert_array_equal(properties, expected_properties)
    assert np.isfinite(properties).all()


def test_glcm_without_pairs_returns_finite_defaults():
    level_image = np.full((3, 3), 3, dtype=np.uint8)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    properties = compute_glcm_properties_for_offset(level_image, mask, 0, 1)
    expected_properties = [0, 0, 0, 0, 0, 1]
    np.testing.assert_array_equal(properties, expected_properties)


def test_lbp_histogram_normalizes_only_valid_centers():
    channel = np.full((5, 5), 64, dtype=np.uint8)
    mask = np.ones((5, 5), dtype=bool)
    histogram = compute_lbp_histogram(channel, mask, 2)
    expected_histogram = np.zeros(10)
    expected_histogram[8] = 1
    np.testing.assert_array_equal(histogram, expected_histogram)
    empty_histogram = compute_lbp_histogram(channel, mask, 3)
    np.testing.assert_array_equal(empty_histogram, np.zeros(10))
    tiny_mask = np.zeros_like(mask)
    tiny_mask[2, 2] = True
    eroded_histogram = compute_lbp_histogram(channel, tiny_mask, 1)
    np.testing.assert_array_equal(eroded_histogram, np.zeros(10))


def test_texture_feature_lengths_and_names_follow_block_order():
    random_source = np.random.default_rng(42)
    rgb = random_source.integers(0, 256, size=(11, 13, 3), dtype=np.uint8)
    mask = np.ones((11, 13), dtype=bool)
    blocks = (
        (compute_lesion_texture_features, list_lesion_texture_names, 26, "lesion_glcm_contrast", "lesion_lbp_r2_09"),
        (compute_leaf_lbp_features, list_leaf_lbp_names, 90, "leaf_lbp_l_r1_00", "leaf_lbp_b_r3_09"),
        (compute_leaf_glcm_features, list_leaf_glcm_names, 54, "leaf_glcm_l_d1_contrast", "leaf_glcm_b_d4_correlation"),
    )
    for compute_features, list_names, length, first, last in blocks:
        features = compute_features(rgb, mask)
        names = list_names()
        assert features.shape == (length,)
        assert features.dtype == np.float64
        assert np.isfinite(features).all()
        assert len(names) == len(set(names)) == length
        assert names[0] == first
        assert names[-1] == last
