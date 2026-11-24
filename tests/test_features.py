from collections import Counter

import numpy as np
import pytest

from coffee_leaf.features.extractor import extract_features, list_feature_names
from coffee_leaf.features.feature_names import describe_feature


def create_feature_inputs():
    random_source = np.random.default_rng(42)
    raw_rgb = random_source.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    processed_rgb = random_source.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    leaf_mask = np.zeros((24, 24), dtype=bool)
    leaf_mask[2:22, 2:22] = True
    lesion_mask = np.zeros_like(leaf_mask)
    lesion_mask[4:8, 5:9] = True
    lesion_mask[15:18, 14:17] = True
    glare_mask = np.zeros_like(leaf_mask)
    glare_mask[4, 5] = True
    return raw_rgb, processed_rgb, leaf_mask, lesion_mask, glare_mask


def test_extract_features_returns_finite_repeatable_vector_without_mutation():
    feature_inputs = create_feature_inputs()
    original_arrays = []
    for input_array in feature_inputs:
        original_arrays.append(input_array.copy())
    first_vector = extract_features(*feature_inputs)
    repeated_vector = extract_features(*feature_inputs)
    assert first_vector.shape == (403,)
    assert first_vector.dtype == np.float64
    assert np.isfinite(first_vector).all()
    np.testing.assert_array_equal(first_vector, repeated_vector)
    for input_array, original_array in zip(feature_inputs, original_arrays):
        np.testing.assert_array_equal(input_array, original_array)


def test_feature_names_have_expected_boundaries_and_descriptions():
    feature_names = list_feature_names()
    assert len(feature_names) == len(set(feature_names)) == 403
    assert feature_names[0] == "lesion_hsv_h_00"
    assert feature_names[134] == "lesion_shape_area_ratio"
    assert feature_names[135] == "leaf_ab_hist_a00_b00"
    assert feature_names[402] == "leaf_glcm_b_d4_correlation"
    region_counts = Counter()
    group_counts = Counter()
    for feature_name in feature_names:
        region, group = describe_feature(feature_name)
        region_counts[region] += 1
        group_counts[group] += 1
    assert region_counts == {"lesion": 135, "leaf": 268}
    assert group_counts == {"color": 202, "texture": 170, "shape": 7, "gradient": 24}


def test_describe_feature_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown feature name"):
        describe_feature("leaf_unknown_value")
    with pytest.raises(ValueError, match="Unknown feature name"):
        describe_feature("unknown_glcm_contrast")


def test_extract_features_for_empty_lesion_keeps_leaf_features():
    raw_rgb, processed_rgb, leaf_mask, lesion_mask, glare_mask = create_feature_inputs()
    empty_lesion_mask = np.zeros_like(lesion_mask)
    features = extract_features(raw_rgb, processed_rgb, leaf_mask, empty_lesion_mask, glare_mask)
    assert not features[:135].any()
    assert features[135:].any()


def test_extract_features_uses_roi_area_after_removing_glare():
    raw_rgb, processed_rgb, leaf_mask, lesion_mask, _ = create_feature_inputs()
    small_lesion_mask = np.zeros_like(lesion_mask)
    clear_glare_mask = np.zeros_like(leaf_mask)
    small_lesion_mask[3, 3:6] = True
    below_threshold = extract_features(raw_rgb, processed_rgb, leaf_mask, small_lesion_mask, clear_glare_mask)
    assert not below_threshold[:135].any()
    small_lesion_mask[3, 6] = True
    at_threshold = extract_features(raw_rgb, processed_rgb, leaf_mask, small_lesion_mask, clear_glare_mask)
    assert at_threshold[:135].any()
    small_lesion_mask[3, 7] = True
    above_threshold = extract_features(raw_rgb, processed_rgb, leaf_mask, small_lesion_mask, clear_glare_mask)
    assert above_threshold[:135].any()
    glare_mask = np.zeros_like(leaf_mask)
    glare_mask[3, 3:5] = True
    after_glare = extract_features(raw_rgb, processed_rgb, leaf_mask, small_lesion_mask, glare_mask)
    assert not after_glare[:135].any()


def test_extract_features_rejects_empty_leaf_and_shape_mismatches():
    raw_rgb, processed_rgb, leaf_mask, lesion_mask, glare_mask = create_feature_inputs()
    empty_leaf_mask = np.zeros_like(leaf_mask)
    wrong_shape_mask = leaf_mask[:-1]
    wrong_shape_rgb = processed_rgb[:-1]
    with pytest.raises(ValueError, match="leaf_mask must contain"):
        extract_features(raw_rgb, processed_rgb, empty_leaf_mask, lesion_mask, glare_mask)
    with pytest.raises(ValueError, match="raw_rgb and processed_rgb must have the same shape"):
        extract_features(raw_rgb, wrong_shape_rgb, leaf_mask, lesion_mask, glare_mask)
    with pytest.raises(ValueError, match="leaf_mask must have shape"):
        extract_features(raw_rgb, processed_rgb, wrong_shape_mask, lesion_mask, glare_mask)
    with pytest.raises(ValueError, match="lesion_mask must have shape"):
        extract_features(raw_rgb, processed_rgb, leaf_mask, wrong_shape_mask, glare_mask)
    with pytest.raises(ValueError, match="glare_mask must have shape"):
        extract_features(raw_rgb, processed_rgb, leaf_mask, lesion_mask, wrong_shape_mask)
