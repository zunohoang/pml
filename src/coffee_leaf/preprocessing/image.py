import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2hsv


GLARE_MIN_VALUE = 0.98
GLARE_MAX_SATURATION = 0.10

WHITE_BALANCE_MIN_GAIN = 0.5
WHITE_BALANCE_MAX_GAIN = 2.0
MIN_CHANNEL_MEAN = 1e-8

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


def load_resized_rgb(image_path, max_side):
    with Image.open(image_path) as image_file:
        oriented_image = ImageOps.exif_transpose(image_file)
        rgb_image = oriented_image.convert("RGB")
        target_size = (max_side, max_side)
        rgb_image.thumbnail(target_size, Image.Resampling.LANCZOS)
        return np.array(rgb_image, dtype=np.uint8)


def find_glare_mask(rgb):
    hsv_image = rgb2hsv(rgb)
    _, saturation, value = np.moveaxis(hsv_image, -1, 0)
    bright_pixel_mask = value >= GLARE_MIN_VALUE
    low_saturation_mask = saturation <= GLARE_MAX_SATURATION
    return bright_pixel_mask & low_saturation_mask


def balance_white(rgb):
    channel_means = rgb.mean(axis=(0, 1))
    target_mean = channel_means.mean()
    safe_channel_means = np.maximum(channel_means, MIN_CHANNEL_MEAN)
    channel_gains = target_mean / safe_channel_means
    bounded_gains = np.clip(channel_gains, WHITE_BALANCE_MIN_GAIN, WHITE_BALANCE_MAX_GAIN)
    balanced_values = rgb * bounded_gains
    rounded_values = np.rint(balanced_values)
    clipped_values = np.clip(rounded_values, 0, 255)
    return clipped_values.astype(np.uint8)


def equalize_lightness(rgb):
    lab_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = lab_image[:, :, 0]
    equalizer = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    equalized_lightness = equalizer.apply(lightness)
    lab_image[:, :, 0] = equalized_lightness
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)


def preprocess_rgb(rgb):
    balanced_rgb = balance_white(rgb)
    return equalize_lightness(balanced_rgb)
