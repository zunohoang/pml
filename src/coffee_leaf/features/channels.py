import numpy as np
from skimage.color import rgb2hsv, rgb2lab


FULL_HUE_DEGREES = 360
GRAY_CHANNEL_WEIGHTS = (0.299, 0.587, 0.114)
MAX_BYTE_VALUE = 255


def validate_rgb_image(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB image must have shape (height, width, 3).")
    if min(rgb.shape[:2]) == 0 or rgb.dtype != np.uint8:
        raise ValueError("RGB image must have positive height and width and dtype uint8.")


def convert_to_hsv(rgb):
    validate_rgb_image(rgb)
    hsv_image = rgb2hsv(rgb)
    hue = hsv_image[:, :, 0]
    hue_degrees = hue * FULL_HUE_DEGREES
    hsv_image[:, :, 0] = hue_degrees
    return hsv_image


def convert_to_lab(rgb):
    validate_rgb_image(rgb)
    return rgb2lab(rgb)


def convert_to_gray(rgb):
    validate_rgb_image(rgb)
    rgb_values = rgb.astype(np.float64)
    channel_weights = np.asarray(GRAY_CHANNEL_WEIGHTS, dtype=np.float64)
    gray_values = rgb_values @ channel_weights
    rounded_values = np.rint(gray_values)
    return rounded_values.astype(np.uint8)


def quantize_channel(channel, low, high):
    low = float(low)
    high = float(high)
    range_width = high - low
    if not np.isfinite(range_width) or range_width <= 0:
        raise ValueError("Channel range must be finite and high must be greater than low.")
    channel_values = np.asarray(channel, dtype=np.float64)
    if not np.isfinite(channel_values).all():
        raise ValueError("Channel values must be finite.")
    bounded_values = np.clip(channel_values, low, high)
    offset_values = bounded_values - low
    channel_fractions = offset_values / range_width
    scaled_values = channel_fractions * MAX_BYTE_VALUE
    rounded_values = np.rint(scaled_values)
    byte_values = np.clip(rounded_values, 0, MAX_BYTE_VALUE)
    return byte_values.astype(np.uint8)
