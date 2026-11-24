import numpy as np

from .channels import convert_to_hsv, convert_to_lab
from .masks import validate_nonempty_mask


LESION_HISTOGRAM_BINS = 16
HSV_HISTOGRAM_CHANNELS = (('h', (0, 360)), ('s', (0, 1)), ('v', (0, 1)))
LAB_HISTOGRAM_CHANNELS = (('l', (0, 100)), ('a', (-128, 128)), ('b', (-128, 128)))

LEAF_HISTOGRAM_BINS = 10
LEAF_CHROMA_RANGES = ((-50, 50), (-30, 70))


def normalize_histogram(counts):
    histogram_values = np.asarray(counts, dtype=np.float64)
    total_count = histogram_values.sum()
    if not np.isfinite(total_count) or np.any(histogram_values < 0):
        raise ValueError("Histogram counts must be finite, nonnegative, and have a finite sum.")
    return histogram_values / max(total_count, 1)


def compute_channel_histograms(color_pixels, histogram_channels):
    channel_histograms = []
    for channel_index, (_, channel_range) in enumerate(histogram_channels):
        channel_values = color_pixels[:, channel_index]
        histogram_counts, _ = np.histogram(channel_values, bins=LESION_HISTOGRAM_BINS, range=channel_range)
        normalized_histogram = normalize_histogram(histogram_counts)
        channel_histograms.append(normalized_histogram)
    return np.concatenate(channel_histograms)


def compute_lesion_color_features(processed_rgb, roi_mask):
    hsv_image = convert_to_hsv(processed_rgb)
    validate_nonempty_mask(roi_mask, "roi_mask", hsv_image.shape[:2])
    lab_image = convert_to_lab(processed_rgb)
    hsv_pixels = hsv_image[roi_mask]
    lab_pixels = lab_image[roi_mask]
    hsv_histograms = compute_channel_histograms(hsv_pixels, HSV_HISTOGRAM_CHANNELS)
    lab_histograms = compute_channel_histograms(lab_pixels, LAB_HISTOGRAM_CHANNELS)
    lab_means = lab_pixels.mean(axis=0)
    lab_deviations = lab_pixels.std(axis=0, ddof=0)
    return np.concatenate((hsv_histograms, lab_histograms, lab_means, lab_deviations))


def compute_leaf_color_features(processed_rgb, leaf_mask):
    lab_image = convert_to_lab(processed_rgb)
    validate_nonempty_mask(leaf_mask, "leaf_mask", lab_image.shape[:2])
    lab_pixels = lab_image[leaf_mask]
    channel_a = lab_pixels[:, 1]
    channel_b = lab_pixels[:, 2]
    histogram_counts, _, _ = np.histogram2d(
        channel_a, channel_b, bins=LEAF_HISTOGRAM_BINS, range=LEAF_CHROMA_RANGES
    )
    flat_histogram = histogram_counts.ravel()
    normalized_histogram = normalize_histogram(flat_histogram)
    return np.sqrt(normalized_histogram)


def list_channel_histogram_names(prefix, histogram_channels):
    feature_names = []
    for channel_name, _ in histogram_channels:
        for bin_index in range(LESION_HISTOGRAM_BINS):
            feature_names.append(f"{prefix}_{channel_name}_{bin_index:02d}")
    return feature_names


def list_lesion_color_names():
    feature_names = list_channel_histogram_names("lesion_hsv", HSV_HISTOGRAM_CHANNELS)
    lab_histogram_names = list_channel_histogram_names("lesion_lab", LAB_HISTOGRAM_CHANNELS)
    feature_names.extend(lab_histogram_names)
    for statistic_name in ("mean", "std"):
        for channel_name, _ in LAB_HISTOGRAM_CHANNELS:
            feature_names.append(f"lesion_lab_{statistic_name}_{channel_name}")
    return feature_names


def list_leaf_color_names():
    feature_names = []
    for a_bin in range(LEAF_HISTOGRAM_BINS):
        for b_bin in range(LEAF_HISTOGRAM_BINS):
            feature_names.append(f"leaf_ab_hist_a{a_bin:02d}_b{b_bin:02d}")
    return feature_names
