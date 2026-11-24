import numpy as np
from scipy.ndimage import binary_erosion


def validate_mask(mask, mask_name, expected_shape=None):
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError(f"{mask_name} must be a two-dimensional boolean array.")
    if min(mask.shape) == 0:
        raise ValueError(f"{mask_name} must have positive height and width.")
    if expected_shape is not None and mask.shape != expected_shape:
        raise ValueError(f"{mask_name} must have shape {expected_shape}, got {mask.shape}.")


def validate_nonempty_mask(mask, mask_name, expected_shape=None):
    validate_mask(mask, mask_name, expected_shape)
    if not mask.any():
        raise ValueError(f"{mask_name} must contain at least one pixel.")


def build_lesion_roi_mask(lesion_mask, leaf_mask, glare_mask):
    validate_mask(leaf_mask, "leaf_mask")
    validate_mask(lesion_mask, "lesion_mask", leaf_mask.shape)
    validate_mask(glare_mask, "glare_mask", leaf_mask.shape)
    lesion_on_leaf_mask = lesion_mask & leaf_mask
    non_glare_mask = ~glare_mask
    return lesion_on_leaf_mask & non_glare_mask


def shrink_mask(mask, window_size):
    validate_mask(mask, "mask")
    if isinstance(window_size, (bool, np.bool_)) or not isinstance(window_size, (int, np.integer)):
        raise ValueError("window_size must be a positive integer.")
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    window_shape = (window_size, window_size)
    square_window = np.ones(window_shape, dtype=bool)
    return binary_erosion(mask, structure=square_window, border_value=0)
