import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from .channels import MAX_BYTE_VALUE, convert_to_gray, convert_to_lab, quantize_channel
from .color import normalize_histogram
from .masks import shrink_mask, validate_mask, validate_nonempty_mask


LBP_NEIGHBOR_COUNT = 8
LBP_BIN_COUNT = 10
LESION_LBP_RADII = (1, 2)
LEAF_LBP_RADII = (1, 2, 3)

GLCM_LEVEL_COUNT = 16
GLCM_LEVEL_WIDTH = (MAX_BYTE_VALUE + 1) // GLCM_LEVEL_COUNT
GLCM_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (-1, 1))
GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation")
EMPTY_GLCM_PROPERTIES = (0, 0, 0, 0, 0, 1)
LESION_GLCM_DISTANCE = 1
LEAF_GLCM_DISTANCES = (1, 2, 4)

LAB_TEXTURE_CHANNELS = (("l", (0, 100)), ("a", (-64, 64)), ("b", (-64, 64)))


def validate_channel(channel):
    if not isinstance(channel, np.ndarray) or channel.ndim != 2 or min(channel.shape) == 0:
        raise ValueError("Texture channel must be a two-dimensional array with positive height and width.")
    if not np.issubdtype(channel.dtype, np.integer):
        raise ValueError("Texture channel must contain integers.")


def validate_positive_integer(value, value_name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{value_name} must be a positive integer.")


def reduce_to_levels(quantized_channel):
    validate_channel(quantized_channel)
    if quantized_channel.dtype != np.uint8:
        raise ValueError("Quantized channel must have dtype uint8.")
    return quantized_channel // GLCM_LEVEL_WIDTH


def compute_lbp_histogram(channel, mask, radius):
    validate_channel(channel)
    validate_mask(mask, "mask", channel.shape)
    validate_positive_integer(radius, "radius")
    window_size = 2 * int(radius) + 1
    if window_size > min(channel.shape):
        return np.zeros(LBP_BIN_COUNT, dtype=np.float64)
    center_mask = shrink_mask(mask, window_size)
    if not center_mask.any():
        return np.zeros(LBP_BIN_COUNT, dtype=np.float64)
    lbp_image = local_binary_pattern(channel, LBP_NEIGHBOR_COUNT, radius, method="uniform")
    center_codes = lbp_image[center_mask].astype(np.intp)
    histogram_counts = np.bincount(center_codes, minlength=LBP_BIN_COUNT)
    return normalize_histogram(histogram_counts)


def get_offset_slices(axis_length, shift):
    if shift < 0:
        return slice(-shift, axis_length), slice(0, axis_length + shift)
    return slice(0, axis_length - shift), slice(shift, axis_length)


def list_valid_pairs(level_image, mask, row_shift, column_shift):
    validate_channel(level_image)
    validate_mask(mask, "mask", level_image.shape)
    if not isinstance(row_shift, (int, np.integer)) or not isinstance(column_shift, (int, np.integer)):
        raise ValueError("row_shift and column_shift must be integers.")
    row_shift = int(row_shift)
    column_shift = int(column_shift)
    height, width = level_image.shape
    if abs(row_shift) >= height or abs(column_shift) >= width:
        return np.empty((0, 2), dtype=level_image.dtype)
    first_rows, second_rows = get_offset_slices(height, row_shift)
    first_columns, second_columns = get_offset_slices(width, column_shift)
    first_mask = mask[first_rows, first_columns]
    second_mask = mask[second_rows, second_columns]
    valid_pair_mask = first_mask & second_mask
    first_pixels = level_image[first_rows, first_columns]
    second_pixels = level_image[second_rows, second_columns]
    return np.column_stack((first_pixels[valid_pair_mask], second_pixels[valid_pair_mask]))


def read_glcm_properties(matrix):
    property_values = []
    for property_name in GLCM_PROPERTIES:
        property_output = graycoprops(matrix, property_name)
        property_value = property_output.item()
        if property_name == "correlation" and not np.isfinite(property_value):
            property_value = 1.0
        property_values.append(property_value)
    return np.array(property_values, dtype=np.float64)


def compute_glcm_properties_for_offset(level_image, mask, row_shift, column_shift):
    level_pairs = list_valid_pairs(level_image, mask, row_shift, column_shift)
    if level_pairs.size == 0:
        return np.array(EMPTY_GLCM_PROPERTIES, dtype=np.float64)
    if level_pairs.min() < 0 or level_pairs.max() >= GLCM_LEVEL_COUNT:
        raise ValueError("GLCM pixel levels must be between 0 and 15.")
    pair_image = level_pairs.astype(np.uint8, copy=False)
    matrix = graycomatrix(pair_image, [1], [0], levels=GLCM_LEVEL_COUNT, symmetric=True, normed=True)
    return read_glcm_properties(matrix)


def compute_glcm_features(level_image, mask, distance):
    validate_positive_integer(distance, "distance")
    direction_features = []
    for row_step, column_step in GLCM_DIRECTIONS:
        row_shift = row_step * int(distance)
        column_shift = column_step * int(distance)
        properties = compute_glcm_properties_for_offset(level_image, mask, row_shift, column_shift)
        direction_features.append(properties)
    return np.mean(direction_features, axis=0)


def compute_lesion_texture_features(processed_rgb, roi_mask):
    gray_image = convert_to_gray(processed_rgb)
    level_image = reduce_to_levels(gray_image)
    glcm_features = compute_glcm_features(level_image, roi_mask, LESION_GLCM_DISTANCE)
    texture_blocks = [glcm_features]
    for radius in LESION_LBP_RADII:
        histogram = compute_lbp_histogram(gray_image, roi_mask, radius)
        texture_blocks.append(histogram)
    return np.concatenate(texture_blocks)


def quantize_lab_texture_channels(raw_rgb, leaf_mask):
    lab_image = convert_to_lab(raw_rgb)
    validate_nonempty_mask(leaf_mask, "leaf_mask", lab_image.shape[:2])
    quantized_channels = []
    for channel_index, (_, channel_range) in enumerate(LAB_TEXTURE_CHANNELS):
        lab_channel = lab_image[:, :, channel_index]
        low, high = channel_range
        quantized_channel = quantize_channel(lab_channel, low, high)
        quantized_channels.append(quantized_channel)
    return quantized_channels


def compute_leaf_lbp_features(raw_rgb, leaf_mask):
    quantized_channels = quantize_lab_texture_channels(raw_rgb, leaf_mask)
    lbp_histograms = []
    for channel in quantized_channels:
        for radius in LEAF_LBP_RADII:
            histogram = compute_lbp_histogram(channel, leaf_mask, radius)
            root_histogram = np.sqrt(histogram)
            lbp_histograms.append(root_histogram)
    return np.concatenate(lbp_histograms)


def compute_leaf_glcm_features(raw_rgb, leaf_mask):
    quantized_channels = quantize_lab_texture_channels(raw_rgb, leaf_mask)
    glcm_blocks = []
    for channel in quantized_channels:
        level_image = reduce_to_levels(channel)
        for distance in LEAF_GLCM_DISTANCES:
            properties = compute_glcm_features(level_image, leaf_mask, distance)
            glcm_blocks.append(properties)
    return np.concatenate(glcm_blocks)


def list_lbp_names(prefix, radii):
    feature_names = []
    for radius in radii:
        for bin_index in range(LBP_BIN_COUNT):
            feature_names.append(f"{prefix}_r{radius}_{bin_index:02d}")
    return feature_names


def list_glcm_names(prefix):
    feature_names = []
    for property_name in GLCM_PROPERTIES:
        lower_name = property_name.lower()
        feature_names.append(f"{prefix}_{lower_name}")
    return feature_names


def list_lesion_texture_names():
    glcm_names = list_glcm_names("lesion_glcm")
    lbp_names = list_lbp_names("lesion_lbp", LESION_LBP_RADII)
    return glcm_names + lbp_names


def list_leaf_lbp_names():
    feature_names = []
    for channel_name, _ in LAB_TEXTURE_CHANNELS:
        channel_names = list_lbp_names(f"leaf_lbp_{channel_name}", LEAF_LBP_RADII)
        feature_names.extend(channel_names)
    return feature_names


def list_leaf_glcm_names():
    feature_names = []
    for channel_name, _ in LAB_TEXTURE_CHANNELS:
        for distance in LEAF_GLCM_DISTANCES:
            prefix = f"leaf_glcm_{channel_name}_d{distance}"
            distance_names = list_glcm_names(prefix)
            feature_names.extend(distance_names)
    return feature_names
