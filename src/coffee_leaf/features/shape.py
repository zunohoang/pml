import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops

from .masks import validate_mask, validate_nonempty_mask


REGION_CONNECTIVITY = 1
SHAPE_FEATURE_NAMES = (
    "lesion_shape_region_count",
    "lesion_shape_mean_area_ratio",
    "lesion_shape_max_area_ratio",
    "lesion_shape_mean_eccentricity",
    "lesion_shape_mean_solidity",
    "lesion_shape_mean_border_distance",
    "lesion_shape_area_ratio",
)


def validate_shape_masks(roi_mask, leaf_mask):
    validate_nonempty_mask(leaf_mask, "leaf_mask")
    validate_mask(roi_mask, "roi_mask", leaf_mask.shape)
    outside_leaf_mask = roi_mask & ~leaf_mask
    if outside_leaf_mask.any():
        raise ValueError("roi_mask must be contained in leaf_mask.")


def compute_region_features(roi_mask, leaf_area):
    region_ids = label(roi_mask, connectivity=REGION_CONNECTIVITY)
    regions = regionprops(region_ids)
    region_areas = []
    region_eccentricities = []
    region_solidities = []
    for region in regions:
        region_areas.append(region.area)
        region_eccentricities.append(region.eccentricity)
        region_solidities.append(region.solidity)
    mean_area_ratio = np.mean(region_areas) / leaf_area
    max_area_ratio = max(region_areas) / leaf_area
    mean_eccentricity = np.mean(region_eccentricities)
    mean_solidity = np.mean(region_solidities)
    region_values = (len(regions), mean_area_ratio, max_area_ratio, mean_eccentricity, mean_solidity)
    return np.array(region_values, dtype=np.float64)


def compute_spatial_features(roi_mask, leaf_mask, leaf_area):
    padded_leaf_mask = np.pad(leaf_mask, 1, mode="constant", constant_values=False)
    padded_distances = distance_transform_edt(padded_leaf_mask)
    leaf_distances = padded_distances[1:-1, 1:-1]
    lesion_distances = leaf_distances[roi_mask]
    leaf_length_scale = np.sqrt(leaf_area)
    normalized_distance = lesion_distances.mean() / leaf_length_scale
    lesion_area = np.count_nonzero(roi_mask)
    area_ratio = lesion_area / leaf_area
    return np.array((normalized_distance, area_ratio), dtype=np.float64)


def compute_shape_features(roi_mask, leaf_mask):
    validate_shape_masks(roi_mask, leaf_mask)
    if not roi_mask.any():
        return np.zeros(len(SHAPE_FEATURE_NAMES), dtype=np.float64)
    leaf_area = np.count_nonzero(leaf_mask)
    region_features = compute_region_features(roi_mask, leaf_area)
    spatial_features = compute_spatial_features(roi_mask, leaf_mask, leaf_area)
    shape_features = np.concatenate((region_features, spatial_features))
    if not np.isfinite(shape_features).all():
        raise ValueError("Shape features must be finite.")
    return shape_features


def list_shape_names():
    return list(SHAPE_FEATURE_NAMES)
