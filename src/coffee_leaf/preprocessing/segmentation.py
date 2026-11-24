from inspect import signature

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2hsv, rgb2lab
from skimage.measure import label
from skimage.morphology import remove_small_objects
from sklearn.cluster import KMeans


MAX_RGB_VALUE = 255
MIN_RGB_SUM = 1e-8

HUE_FULL_CIRCLE = 360
LEAF_MIN_HUE = 8
LEAF_MAX_HUE = 170
LEAF_MIN_SATURATION = 0.15
LEAF_MIN_VALUE = 0.06
LEAF_MIN_SEED_EXCESS_GREEN = 0.04

LEAF_KERNEL_SIZE = 3
FOUR_CONNECTIVITY = 1
BACKGROUND_REGION_ID = 0
LEAF_MIN_PIXELS = 40

LESION_CLUSTER_COUNT = 3
LESION_KMEANS_RUNS = 10
LESION_KMEANS_SEED = 42
LESION_MIN_COLOR_DISTANCE = 8

LESION_MIN_REGION_PIXELS = 3
SMALL_OBJECT_PARAMETERS = signature(remove_small_objects).parameters


def calculate_excess_green(raw_rgb):
    rgb_values = raw_rgb.astype(float)
    scaled_values = rgb_values / MAX_RGB_VALUE
    channel_sums = scaled_values.sum(axis=-1, keepdims=True)
    safe_channel_sums = np.maximum(channel_sums, MIN_RGB_SUM)
    normalized_rgb = scaled_values / safe_channel_sums
    normalized_channels = np.moveaxis(normalized_rgb, -1, 0)
    red = normalized_channels[0]
    green = normalized_channels[1]
    blue = normalized_channels[2]
    return 2 * green - red - blue


def find_leaf_color_candidates(hsv_image):
    hsv_channels = np.moveaxis(hsv_image, -1, 0)
    hue = hsv_channels[0]
    saturation = hsv_channels[1]
    value = hsv_channels[2]
    hue_degrees = hue * HUE_FULL_CIRCLE
    hue_range_mask = (hue_degrees >= LEAF_MIN_HUE) & (hue_degrees <= LEAF_MAX_HUE)
    saturated_pixel_mask = saturation > LEAF_MIN_SATURATION
    bright_pixel_mask = value > LEAF_MIN_VALUE
    return hue_range_mask & saturated_pixel_mask & bright_pixel_mask


def find_leaf_seed_mask(excess_green, hsv_image):
    hsv_channels = np.moveaxis(hsv_image, -1, 0)
    saturation = hsv_channels[1]
    green_pixel_mask = excess_green > LEAF_MIN_SEED_EXCESS_GREEN
    saturated_pixel_mask = saturation > LEAF_MIN_SATURATION
    return green_pixel_mask & saturated_pixel_mask


def clean_leaf_candidates(candidate_mask):
    kernel_shape = (LEAF_KERNEL_SIZE, LEAF_KERNEL_SIZE)
    kernel = np.ones(kernel_shape, dtype=np.uint8)
    candidate_values = candidate_mask.astype(np.uint8)
    opened_mask = cv2.morphologyEx(candidate_values, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    return closed_mask.astype(bool)


def select_leaf_region(candidate_mask, seed_mask):
    region_ids = label(candidate_mask, connectivity=FOUR_CONNECTIVITY)
    region_count = int(region_ids.max()) + 1
    seed_region_ids = region_ids[seed_mask]
    region_scores = np.bincount(seed_region_ids, minlength=region_count)
    region_scores[BACKGROUND_REGION_ID] = 0
    if not region_scores.any():
        flat_region_ids = region_ids.ravel()
        region_scores = np.bincount(flat_region_ids)
        region_scores[BACKGROUND_REGION_ID] = 0
    selected_region_id = region_scores.argmax()
    if selected_region_id == BACKGROUND_REGION_ID:
        return candidate_mask
    return region_ids == selected_region_id


def validate_leaf_area(leaf_mask):
    leaf_pixel_count = leaf_mask.sum()
    if leaf_pixel_count < LEAF_MIN_PIXELS:
        raise ValueError(f"Leaf mask contains fewer than {LEAF_MIN_PIXELS} pixels.")


def find_leaf_mask(raw_rgb):
    hsv_image = rgb2hsv(raw_rgb)
    excess_green = calculate_excess_green(raw_rgb)
    candidate_mask = find_leaf_color_candidates(hsv_image)
    seed_mask = find_leaf_seed_mask(excess_green, hsv_image)
    cleaned_candidate_mask = clean_leaf_candidates(candidate_mask)
    leaf_region_mask = select_leaf_region(cleaned_candidate_mask, seed_mask)
    leaf_mask = binary_fill_holes(leaf_region_mask)
    validate_leaf_area(leaf_mask)
    return leaf_mask


def extract_lesion_chroma(processed_rgb, lesion_roi_mask):
    lab_image = rgb2lab(processed_rgb)
    roi_lab_values = lab_image[lesion_roi_mask]
    roi_chroma_values = roi_lab_values[:, 1:]
    return roi_chroma_values


def cluster_leaf_colors(chroma_values):
    color_model = KMeans(
        n_clusters=LESION_CLUSTER_COUNT,
        n_init=LESION_KMEANS_RUNS,
        random_state=LESION_KMEANS_SEED,
    )
    return color_model.fit(chroma_values)


def select_lesion_cluster(roi_cluster_ids, cluster_centers):
    pixel_counts_per_cluster = np.bincount(roi_cluster_ids, minlength=LESION_CLUSTER_COUNT)
    reference_cluster_id = pixel_counts_per_cluster.argmax()
    reference_center = cluster_centers[reference_cluster_id]
    center_differences = cluster_centers - reference_center
    distances_from_reference = np.linalg.norm(center_differences, axis=1)
    empty_cluster_mask = pixel_counts_per_cluster == 0
    distances_from_reference[empty_cluster_mask] = 0
    farthest_cluster_id = distances_from_reference.argmax()
    farthest_cluster_distance = distances_from_reference[farthest_cluster_id]
    if farthest_cluster_distance < LESION_MIN_COLOR_DISTANCE:
        return None
    return farthest_cluster_id


def remove_tiny_lesions(lesion_mask):
    if "max_size" in SMALL_OBJECT_PARAMETERS:
        largest_removed_size = LESION_MIN_REGION_PIXELS - 1
        return remove_small_objects(lesion_mask, max_size=largest_removed_size, connectivity=FOUR_CONNECTIVITY)
    return remove_small_objects(lesion_mask, min_size=LESION_MIN_REGION_PIXELS, connectivity=FOUR_CONNECTIVITY)


def find_lesion_mask(processed_rgb, leaf_mask, glare_mask):
    lesion_roi_mask = leaf_mask & ~glare_mask
    if lesion_roi_mask.sum() < LESION_CLUSTER_COUNT:
        return np.zeros(leaf_mask.shape, dtype=bool)
    roi_chroma_values = extract_lesion_chroma(processed_rgb, lesion_roi_mask)
    color_model = cluster_leaf_colors(roi_chroma_values)
    lesion_cluster_id = select_lesion_cluster(color_model.labels_, color_model.cluster_centers_)
    lesion_mask = np.zeros(leaf_mask.shape, dtype=bool)
    if lesion_cluster_id is None:
        return lesion_mask
    lesion_mask[lesion_roi_mask] = color_model.labels_ == lesion_cluster_id
    return remove_tiny_lesions(lesion_mask)
