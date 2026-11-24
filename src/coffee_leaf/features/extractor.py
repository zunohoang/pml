import numpy as np

from ..preprocessing.image import find_glare_mask, load_resized_rgb, preprocess_rgb
from ..preprocessing.segmentation import find_leaf_mask, find_lesion_mask
from .channels import validate_rgb_image
from .color import compute_leaf_color_features, compute_lesion_color_features, list_leaf_color_names, list_lesion_color_names
from .gradient import compute_gradient_features, list_gradient_names
from .masks import build_lesion_roi_mask, validate_nonempty_mask
from .shape import compute_shape_features, list_shape_names
from .texture import (
    compute_leaf_glcm_features,
    compute_leaf_lbp_features,
    compute_lesion_texture_features,
    list_leaf_glcm_names,
    list_leaf_lbp_names,
    list_lesion_texture_names,
)


LESION_MIN_AREA_RATIO = 0.01
LESION_FEATURE_COUNT = 135
EXPECTED_FEATURE_COUNT = 403


def list_feature_names():
    feature_names = list_lesion_color_names()
    feature_names.extend(list_lesion_texture_names())
    feature_names.extend(list_shape_names())
    feature_names.extend(list_leaf_color_names())
    feature_names.extend(list_leaf_lbp_names())
    feature_names.extend(list_gradient_names())
    feature_names.extend(list_leaf_glcm_names())
    return feature_names


def validate_feature_inputs(raw_rgb, processed_rgb, leaf_mask):
    validate_rgb_image(raw_rgb)
    validate_rgb_image(processed_rgb)
    if raw_rgb.shape != processed_rgb.shape:
        raise ValueError("raw_rgb and processed_rgb must have the same shape.")
    validate_nonempty_mask(leaf_mask, "leaf_mask", raw_rgb.shape[:2])


def compute_lesion_features(processed_rgb, roi_mask, leaf_mask):
    leaf_area = np.count_nonzero(leaf_mask)
    lesion_area = np.count_nonzero(roi_mask)
    lesion_area_ratio = lesion_area / leaf_area
    if lesion_area_ratio < LESION_MIN_AREA_RATIO:
        return np.zeros(LESION_FEATURE_COUNT, dtype=np.float64)
    color_features = compute_lesion_color_features(processed_rgb, roi_mask)
    texture_features = compute_lesion_texture_features(processed_rgb, roi_mask)
    shape_features = compute_shape_features(roi_mask, leaf_mask)
    return np.concatenate((color_features, texture_features, shape_features))


def compute_leaf_features(raw_rgb, processed_rgb, leaf_mask):
    color_features = compute_leaf_color_features(processed_rgb, leaf_mask)
    lbp_features = compute_leaf_lbp_features(raw_rgb, leaf_mask)
    gradient_features = compute_gradient_features(raw_rgb, leaf_mask)
    glcm_features = compute_leaf_glcm_features(raw_rgb, leaf_mask)
    return np.concatenate((color_features, lbp_features, gradient_features, glcm_features))


def validate_feature_vector(feature_vector):
    if feature_vector.shape != (EXPECTED_FEATURE_COUNT,):
        raise ValueError(f"Feature vector must have shape ({EXPECTED_FEATURE_COUNT},).")
    if not np.isfinite(feature_vector).all():
        raise ValueError("Feature vector must contain only finite values.")


def extract_features(raw_rgb, processed_rgb, leaf_mask, lesion_mask, glare_mask):
    validate_feature_inputs(raw_rgb, processed_rgb, leaf_mask)
    roi_mask = build_lesion_roi_mask(lesion_mask, leaf_mask, glare_mask)
    lesion_features = compute_lesion_features(processed_rgb, roi_mask, leaf_mask)
    leaf_features = compute_leaf_features(raw_rgb, processed_rgb, leaf_mask)
    feature_vector = np.concatenate((lesion_features, leaf_features))
    validate_feature_vector(feature_vector)
    return feature_vector


def extract_features_from_path(image_path, max_side):
    raw_rgb = load_resized_rgb(image_path, max_side)
    glare_mask = find_glare_mask(raw_rgb)
    processed_rgb = preprocess_rgb(raw_rgb)
    leaf_mask = find_leaf_mask(raw_rgb)
    lesion_mask = find_lesion_mask(processed_rgb, leaf_mask, glare_mask)
    return extract_features(raw_rgb, processed_rgb, leaf_mask, lesion_mask, glare_mask)
