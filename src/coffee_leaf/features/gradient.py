import numpy as np

from .channels import convert_to_lab
from .color import normalize_histogram
from .masks import shrink_mask, validate_nonempty_mask


GRADIENT_WINDOW_SIZE = 3
GRADIENT_HISTOGRAM_BINS = 8
GRADIENT_CHANNELS = (('l', (0, 30)), ('a', (0, 20)), ('b', (0, 20)))


def compute_gradient_histograms(lab_image, inner_leaf_mask):
    channel_histograms = []
    for channel_index, (_, gradient_range) in enumerate(GRADIENT_CHANNELS):
        channel_values = lab_image[:, :, channel_index]
        row_gradient, column_gradient = np.gradient(channel_values)
        gradient_magnitude = np.hypot(row_gradient, column_gradient)
        leaf_gradients = gradient_magnitude[inner_leaf_mask]
        histogram_counts, _ = np.histogram(leaf_gradients, bins=GRADIENT_HISTOGRAM_BINS, range=gradient_range)
        normalized_histogram = normalize_histogram(histogram_counts)
        gradient_histogram = np.sqrt(normalized_histogram)
        channel_histograms.append(gradient_histogram)
    return np.concatenate(channel_histograms)


def compute_gradient_features(raw_rgb, leaf_mask):
    lab_image = convert_to_lab(raw_rgb)
    image_shape = lab_image.shape[:2]
    validate_nonempty_mask(leaf_mask, "leaf_mask", image_shape)
    if min(image_shape) < 2:
        raise ValueError("Gradient features require at least two pixels per image axis.")
    inner_leaf_mask = shrink_mask(leaf_mask, GRADIENT_WINDOW_SIZE)
    return compute_gradient_histograms(lab_image, inner_leaf_mask)


def list_gradient_names():
    feature_names = []
    for channel_name, _ in GRADIENT_CHANNELS:
        for bin_index in range(GRADIENT_HISTOGRAM_BINS):
            feature_names.append(f"leaf_grad_{channel_name}_{bin_index:02d}")
    return feature_names
